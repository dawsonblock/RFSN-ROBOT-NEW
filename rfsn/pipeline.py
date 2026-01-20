"""
V12 Control Pipeline: Composable Control Architecture
======================================================
Decomposes the monolithic RFSNHarness into pluggable components:
- Observer: Builds ObsPacket from simulation state
- Executive: Wraps state machine + learner for decision making
- SafetyManager: Enforces safety constraints
- Controller: Computes torques (PD, MPC, Impedance)
- Logger: Structured logging

ControlPipeline orchestrates these components.
"""

import mujoco as mj
import numpy as np
from typing import Optional, Protocol, Dict, Any
from dataclasses import dataclass

from .obs_packet import ObsPacket
from .decision import RFSNDecision


class ObserverProtocol(Protocol):
    """Protocol for observation building."""
    def observe(self, model: mj.MjModel, data: mj.MjData, t: float, dt: float) -> ObsPacket: ...


class ExecutiveProtocol(Protocol):
    """Protocol for decision making."""
    def decide(self, obs: ObsPacket) -> RFSNDecision: ...
    def reset(self) -> None: ...


class SafetyManagerProtocol(Protocol):
    """Protocol for safety enforcement."""
    def check_events(self, obs: ObsPacket) -> Optional[str]: ...
    def enforce(self, decision: RFSNDecision, obs: ObsPacket) -> RFSNDecision: ...


class ControllerProtocol(Protocol):
    """Protocol for torque computation."""
    def compute_torques(self, model: mj.MjModel, data: mj.MjData, 
                       obs: ObsPacket, decision: RFSNDecision) -> np.ndarray: ...


class LoggerProtocol(Protocol):
    """Protocol for logging."""
    def log_step(self, obs: ObsPacket, decision: RFSNDecision, tau: np.ndarray) -> None: ...
    def log_event(self, event_type: str, data: Dict[str, Any]) -> None: ...


@dataclass
class PipelineConfig:
    """Configuration for control pipeline."""
    task_name: str = "pick_place"
    controller_type: str = "joint_mpc"  # pd, joint_mpc, task_mpc, impedance
    enable_learning: bool = False
    enable_logging: bool = True
    planning_interval: int = 5  # Steps between MPC replans


class MujocoObserver:
    """
    V12 Observer: Builds ObsPacket from MuJoCo state.
    
    Extracts all relevant state information and contact signals.
    """
    
    def __init__(self, model: mj.MjModel, task_name: str = "pick_place"):
        """Initialize observer with model reference."""
        self.model = model
        self.task_name = task_name
        
        # Cache body/geom IDs
        from .mujoco_utils import init_id_cache, get_id_cache
        init_id_cache(model)
        self.ids = get_id_cache()
    
    def observe(self, model: mj.MjModel, data: mj.MjData, t: float, dt: float) -> ObsPacket:
        """Build observation packet from current simulation state."""
        from .mujoco_utils import build_obs_packet
        return build_obs_packet(model, data, t=t, dt=dt, task_name=self.task_name)


class RFSNExecutive:
    """
    V12 Executive: Wraps state machine and learner for decision making.
    
    Coordinates profile selection (via learner) and state transitions.
    """
    
    def __init__(self, state_machine, learner=None, profile_library=None):
        """
        Initialize executive.
        
        Args:
            state_machine: RFSNStateMachine instance
            learner: Optional learner (ContextualProfileLearner or SafeLearner)
            profile_library: ProfileLibraryV2 instance
        """
        self.sm = state_machine
        self.learner = learner
        self.profile_library = profile_library
        self.grasp_quality_cache = None
    
    def decide(self, obs: ObsPacket, grasp_quality: Optional[dict] = None) -> RFSNDecision:
        """
        Generate decision for current observation.
        
        Args:
            obs: Current observation
            grasp_quality: Optional grasp quality metrics
            
        Returns:
            Decision with target pose and control parameters
        """
        profile_override = None
        
        if self.learner:
            # Use learner to select profile
            profile_override = self.learner.select_profile(obs, self.sm.current_state)
        
        # Store grasp quality for use by state machine
        self.grasp_quality_cache = grasp_quality
        
        # Get decision from state machine
        decision = self.sm.step(obs, profile_override=profile_override,
                               grasp_quality=grasp_quality)
        
        return decision
    
    def update_learner(self, transition: dict):
        """Update learner with transition data."""
        if self.learner and hasattr(self.learner, 'update'):
            self.learner.update(transition)
    
    def reset(self):
        """Reset executive state."""
        self.sm.reset()
        self.grasp_quality_cache = None
    
    @property
    def current_state(self) -> str:
        """Get current state from state machine."""
        return self.sm.current_state


class SafetyManagerV2:
    """
    V12 Safety Manager: Encapsulates all safety logic.
    
    Two-phase safety:
    1. check_events(): Detect safety violations
    2. enforce(): Modify decision to maintain safety
    """
    
    def __init__(self, config: dict = None):
        """Initialize safety manager with bounds."""
        config = config or {}
        
        # MPC parameter bounds
        self.H_min = config.get('H_min', 5)
        self.H_max = config.get('H_max', 30)
        self.Q_min = config.get('Q_min', 1.0)
        self.Q_max = config.get('Q_max', 500.0)
        self.R_min = config.get('R_min', 0.001)
        self.R_max = config.get('R_max', 0.5)
        self.du_min = config.get('du_min', 0.001)
        self.du_max = config.get('du_max', 0.5)
        self.tau_scale_min = config.get('tau_scale_min', 0.1)
        self.tau_scale_max = config.get('tau_scale_max', 1.0)
        
        # Thresholds
        self.penetration_threshold = config.get('penetration_threshold', 0.05)
        self.mpc_fail_threshold = config.get('mpc_fail_threshold', 3)
        self.torque_sat_threshold = config.get('torque_sat_threshold', 5)
        
        # State tracking
        self.mpc_fail_count = 0
        self.poison_list = set()
        self.recover_count = 0
        self.last_severe_event = None
    
    def check_events(self, obs: ObsPacket) -> Optional[str]:
        """
        Check for safety violations.
        
        Returns:
            Description of safety event, or None if safe
        """
        if obs.self_collision:
            return "self_collision"
        
        if obs.table_collision:
            return "table_collision"
        
        if obs.penetration > self.penetration_threshold:
            return f"penetration_{obs.penetration:.4f}m"
        
        if self.mpc_fail_count >= self.mpc_fail_threshold:
            return f"mpc_nonconvergence_{self.mpc_fail_count}"
        
        if obs.torque_sat_count >= self.torque_sat_threshold:
            return f"torque_saturation_{obs.torque_sat_count}"
        
        if obs.joint_limit_proximity > 0.98:
            return f"joint_limit_{obs.joint_limit_proximity:.2f}"
        
        return None
    
    def enforce(self, decision: RFSNDecision, obs: ObsPacket) -> RFSNDecision:
        """
        Enforce safety constraints on decision.
        
        Args:
            decision: Proposed decision
            obs: Current observation
            
        Returns:
            Safety-constrained decision
        """
        # Check for severe events
        severe_event = self.check_events(obs)
        
        if severe_event:
            self.last_severe_event = severe_event
            self.recover_count += 1
            decision = self._create_recover_decision(obs, severe_event)
            print(f"[SAFETY_V2] Forcing RECOVER: {severe_event}")
        
        # Clamp MPC parameters
        decision.horizon_steps = int(np.clip(decision.horizon_steps, self.H_min, self.H_max))
        decision.Q_diag = np.clip(decision.Q_diag, self.Q_min, self.Q_max)
        decision.R_diag = np.clip(decision.R_diag, self.R_min, self.R_max)
        decision.terminal_Q_diag = np.clip(decision.terminal_Q_diag, self.Q_min, self.Q_max)
        decision.du_penalty = float(np.clip(decision.du_penalty, self.du_min, self.du_max))
        decision.max_tau_scale = float(np.clip(decision.max_tau_scale, self.tau_scale_min, self.tau_scale_max))
        
        # Track MPC convergence
        if not obs.mpc_converged:
            self.mpc_fail_count += 1
        else:
            self.mpc_fail_count = 0
        
        return decision
    
    def _create_recover_decision(self, obs: ObsPacket, reason: str) -> RFSNDecision:
        """Create conservative RECOVER decision."""
        x_safe = obs.x_ee_pos.copy()
        x_safe[2] += 0.05  # Up 5cm
        
        return RFSNDecision(
            task_mode="RECOVER",
            x_target_pos=x_safe,
            x_target_quat=obs.x_ee_quat.copy(),
            horizon_steps=self.H_min,
            Q_diag=np.array([60.0] * 7 + [12.0] * 7),
            R_diag=0.05 * np.ones(7),
            terminal_Q_diag=np.array([120.0] * 7 + [24.0] * 7),
            du_penalty=0.05,
            max_tau_scale=0.4,
            contact_policy="AVOID",
            confidence=0.0,
            reason=f"RECOVER: {reason}",
            rollback_token="safety_forced"
        )
    
    def poison_profile(self, state: str, profile: str):
        """Add profile to poison list."""
        self.poison_list.add((state, profile))
    
    def is_poisoned(self, state: str, profile: str) -> bool:
        """Check if profile is poisoned."""
        return (state, profile) in self.poison_list


class ControllerFactory:
    """Factory for creating controller instances."""
    
    @staticmethod
    def create(controller_type: str, model: mj.MjModel, config: dict = None):
        """
        Create controller of specified type.
        
        Args:
            controller_type: 'pd', 'joint_mpc', 'task_mpc', 'impedance'
            model: MuJoCo model
            config: Optional controller configuration
            
        Returns:
            Controller instance
        """
        config = config or {}

        # Normalize controller type to canonical form.  This ensures aliases
        # map to the correct controller implementation and avoids subtle
        # mismatches between strings such as "task_space_mpc" vs "task_mpc".
        from .controller_types import normalize_controller_type
        ctype = normalize_controller_type(controller_type)

        if ctype == "pd":
            return PDControllerV2(model, config)
        elif ctype == "joint_mpc":
            return JointMPCControllerV2(model, config)
        elif ctype == "task_mpc":
            return TaskSpaceMPCControllerV2(model, config)
        elif ctype == "impedance":
            return ImpedanceControllerV2(model, config)
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")


class PDControllerV2:
    """V12 PD Controller with inverse dynamics."""
    
    def __init__(self, model: mj.MjModel, config: dict = None):
        config = config or {}
        self.model = model
        
        # Base PD gains
        self.KP = np.array([300.0, 300.0, 300.0, 300.0, 150.0, 100.0, 50.0])
        self.KD = np.array([60.0, 60.0, 60.0, 60.0, 30.0, 20.0, 10.0])
        
        # Temp data for inverse dynamics
        self._temp_data = None
    
    def compute_torques(self, model: mj.MjModel, data: mj.MjData,
                       obs: ObsPacket, decision: RFSNDecision,
                       q_target: np.ndarray = None,
                       qd_target: np.ndarray = None) -> np.ndarray:
        """Compute PD control torques using inverse dynamics."""
        if q_target is None:
            q_target = obs.q  # Hold position
        if qd_target is None:
            qd_target = np.zeros(7)
        
        # Scale gains based on profile Q_diag (legacy compatibility)
        kp_scale = np.sqrt(decision.Q_diag[:7] / 50.0)
        kd_scale = np.sqrt(decision.Q_diag[7:14] / 10.0)
        KP_local = self.KP * kp_scale
        KD_local = self.KD * kd_scale
        
        # PD error
        q_error = q_target - obs.q
        qd_error = qd_target - obs.qd
        
        # Compute desired acceleration
        qacc = KP_local * q_error + KD_local * qd_error
        
        # Inverse dynamics
        if self._temp_data is None:
            self._temp_data = mj.MjData(model)
        
        self._temp_data.qpos[:] = data.qpos
        self._temp_data.qvel[:] = data.qvel
        self._temp_data.qacc[:] = 0
        self._temp_data.qacc[:7] = qacc
        
        mj.mj_inverse(model, self._temp_data)
        tau = self._temp_data.qfrc_inverse[:7].copy()
        
        # Apply torque scale and saturate
        tau *= decision.max_tau_scale
        tau = np.clip(tau, -87.0, 87.0)
        
        return tau


class JointMPCControllerV2:
    """V12 Joint-Space MPC Controller with anytime behavior."""
    
    def __init__(self, model: mj.MjModel, config: dict = None):
        config = config or {}
        self.model = model
        
        # Planning cadence
        self.planning_interval = config.get('planning_interval', 5)
        self.step_count = 0
        self.last_plan_step = -999
        
        # Cached trajectory
        self.cached_q_ref = None
        self.cached_qd_ref = None
        
        # Fallback controller
        self.pd_fallback = PDControllerV2(model, config)
        
        # MPC solver
        self.mpc_solver = None
        self.last_result = None
        self._init_solver(config)
    
    def _init_solver(self, config: dict):
        """Initialize MPC solver."""
        try:
            from .mpc_receding import RecedingHorizonMPCQP, MPCConfig
            mpc_config = MPCConfig(
                H_min=config.get('H_min', 5),
                H_max=config.get('H_max', 30),
                max_iterations=config.get('max_iterations', 100),
                time_budget_ms=config.get('time_budget_ms', 50.0),
                learning_rate=0.1,
                warm_start=True
            )
            self.mpc_solver = RecedingHorizonMPCQP(mpc_config)
        except ImportError:
            print("[CONTROLLER_V2] MPC solver not available, using PD fallback")
            self.mpc_solver = None
    
    def compute_torques(self, model: mj.MjModel, data: mj.MjData,
                       obs: ObsPacket, decision: RFSNDecision,
                       q_target: np.ndarray = None,
                       qd_target: np.ndarray = None) -> np.ndarray:
        """Compute MPC control torques with anytime fallback.

        A valid q_target must always be supplied.  The control pipeline
        is responsible for calling decision_to_joint_target() to provide
        this value.  Falling back to hold position is disallowed to
        prevent silent failures where the robot does not move.
        """
        if q_target is None:
            raise RuntimeError(
                "JointMPCControllerV2 requires q_target; the pipeline must provide "
                "a joint target via decision_to_joint_target()"
            )
        
        self.step_count += 1
        
        # Check if we should replan
        should_replan = (self.step_count - self.last_plan_step) >= self.planning_interval
        
        if should_replan and self.mpc_solver is not None:
            result = self._solve_mpc(obs, decision, q_target)
            
            if result is not None and (result.converged or result.reason == "max_iters"):
                self.cached_q_ref = result.q_ref_next
                self.cached_qd_ref = result.qd_ref_next
                self.last_plan_step = self.step_count
                self.last_result = result
                
                obs.controller_mode = "MPC_TRACKING"
                obs.mpc_converged = result.converged
                obs.mpc_solve_time_ms = result.solve_time_ms
            else:
                # MPC failed, use fallback
                obs.controller_mode = "ID_SERVO"
                obs.fallback_used = True
                obs.mpc_failure_reason = result.reason if result else "solver_error"
        
        # Use cached reference or fallback
        if self.cached_q_ref is not None:
            q_ref = self.cached_q_ref
            qd_ref = self.cached_qd_ref
        else:
            q_ref = q_target
            qd_ref = np.zeros(7)
        
        # Compute torques using PD tracking
        return self.pd_fallback.compute_torques(
            model, data, obs, decision, q_target=q_ref, qd_target=qd_ref
        )
    
    def _solve_mpc(self, obs: ObsPacket, decision: RFSNDecision, q_target: np.ndarray):
        """Solve MPC problem."""
        try:
            mpc_params = {
                'horizon_steps': decision.horizon_steps,
                'Q_diag': decision.Q_diag,
                'R_diag': decision.R_diag,
                'terminal_Q_diag': decision.terminal_Q_diag,
                'du_penalty': decision.du_penalty,
                'joint_limit_proximity': obs.joint_limit_proximity
            }
            
            joint_limits = (self.model.jnt_range[:7, 0], self.model.jnt_range[:7, 1])
            
            return self.mpc_solver.solve(
                obs.q, obs.qd, q_target, 
                self.model.opt.timestep, mpc_params, joint_limits
            )
        except Exception as e:
            print(f"[MPC_V2] Solve error: {e}")
            return None
    
    def reset(self):
        """Reset controller state."""
        self.step_count = 0
        self.last_plan_step = -999
        self.cached_q_ref = None
        self.cached_qd_ref = None
        if self.mpc_solver:
            self.mpc_solver.reset_warm_start()


class TaskSpaceMPCControllerV2:
    """V12 Task-Space MPC Controller."""
    
    def __init__(self, model: mj.MjModel, config: dict = None):
        config = config or {}
        self.model = model
        self.planning_interval = config.get('planning_interval', 5)
        self.step_count = 0
        self.last_plan_step = -999
        self.cached_q_ref = None
        self.cached_qd_ref = None
        self.pd_fallback = PDControllerV2(model, config)
        
        # Task-space solver
        self.ts_solver = None
        self._init_solver(config)
    
    def _init_solver(self, config: dict):
        """Initialize task-space MPC solver."""
        try:
            from .mpc_task_space import TaskSpaceRecedingHorizonMPC, TaskSpaceMPCConfig
            ts_config = TaskSpaceMPCConfig(
                H_min=config.get('H_min', 5),
                H_max=config.get('H_max', 30),
                max_iterations=config.get('max_iterations', 100),
                time_budget_ms=config.get('time_budget_ms', 50.0),
            )
            self.ts_solver = TaskSpaceRecedingHorizonMPC(self.model, ts_config)
        except ImportError:
            print("[CONTROLLER_V2] Task-space MPC solver not available")
            self.ts_solver = None
    
    def compute_torques(self, model: mj.MjModel, data: mj.MjData,
                       obs: ObsPacket, decision: RFSNDecision,
                       q_target: np.ndarray = None,
                       qd_target: np.ndarray = None) -> np.ndarray:
        """Compute task-space MPC control torques."""
        self.step_count += 1
        should_replan = (self.step_count - self.last_plan_step) >= self.planning_interval
        
        if should_replan and self.ts_solver is not None:
            result = self._solve_task_mpc(obs, decision)
            
            if result is not None and (result.converged or result.reason == "max_iters"):
                self.cached_q_ref = result.q_ref_next
                self.cached_qd_ref = result.qd_ref_next
                self.last_plan_step = self.step_count
                
                obs.controller_mode = "TASK_SPACE_MPC"
                obs.mpc_converged = result.converged
                obs.mpc_solve_time_ms = result.solve_time_ms
        
        # Use cached or fallback
        q_ref = self.cached_q_ref if self.cached_q_ref is not None else obs.q
        qd_ref = self.cached_qd_ref if self.cached_qd_ref is not None else np.zeros(7)
        
        return self.pd_fallback.compute_torques(
            model, data, obs, decision, q_target=q_ref, qd_target=qd_ref
        )
    
    def _solve_task_mpc(self, obs: ObsPacket, decision: RFSNDecision):
        """Solve task-space MPC problem."""
        try:
            ts_params = {
                'horizon_steps': decision.horizon_steps,
                'Q_pos_task': decision.Q_diag[:3],
                'Q_ori_task': decision.Q_diag[3:6] * 0.1,
                'Q_vel_task': decision.Q_diag[7:13],
                'R_diag': decision.R_diag,
                'terminal_Q_pos': decision.terminal_Q_diag[:3],
                'terminal_Q_ori': decision.terminal_Q_diag[3:6] * 0.1,
                'du_penalty': decision.du_penalty
            }
            
            return self.ts_solver.solve(
                obs.q, obs.qd,
                decision.x_target_pos, decision.x_target_quat,
                self.model.opt.timestep, ts_params
            )
        except Exception as e:
            print(f"[TS_MPC_V2] Solve error: {e}")
            return None
    
    def reset(self):
        """Reset controller state."""
        self.step_count = 0
        self.last_plan_step = -999
        self.cached_q_ref = None
        self.cached_qd_ref = None
        if self.ts_solver:
            self.ts_solver.reset_warm_start()


class ImpedanceControllerV2:
    """V12 Impedance Controller wrapper."""
    
    def __init__(self, model: mj.MjModel, config: dict = None):
        config = config or {}
        self.model = model
        
        # Try to use existing impedance controller
        try:
            from .impedance_controller import ImpedanceController
            self.impl = ImpedanceController(model)
        except ImportError:
            self.impl = None
            print("[CONTROLLER_V2] Impedance controller not available")
        
        self.pd_fallback = PDControllerV2(model, config)
    
    def compute_torques(self, model: mj.MjModel, data: mj.MjData,
                       obs: ObsPacket, decision: RFSNDecision,
                       q_target: np.ndarray = None,
                       qd_target: np.ndarray = None) -> np.ndarray:
        """Compute impedance control torques."""
        if self.impl is None:
            return self.pd_fallback.compute_torques(
                model, data, obs, decision, q_target, qd_target
            )
        
        # Prepare force signals
        force_signals = {
            'ee_table_fN': obs.ee_table_fN,
            'cube_table_fN': obs.cube_table_fN,
            'cube_fingers_fN': obs.cube_fingers_fN,
            'force_signal_is_proxy': obs.force_signal_is_proxy
        }
        
        tau = self.impl.compute_torques(
            data,
            decision.x_target_pos,
            decision.x_target_quat,
            nullspace_target_q=q_target,
            force_signals=force_signals,
            state_name=decision.task_mode
        )
        
        tau *= decision.max_tau_scale
        tau = np.clip(tau, -87.0, 87.0)
        
        obs.controller_mode = "IMPEDANCE"
        return tau
    
    def reset(self):
        """Reset controller state."""
        pass


class ControlPipeline:
    """
    V12 Control Pipeline: Orchestrates all components.
    
    Single entry point for control loop that coordinates:
    - Observation building
    - Decision making (state machine + learning)
    - Safety enforcement
    - Torque computation
    - Logging
    """
    
    def __init__(self, observer: ObserverProtocol,
                 executive: ExecutiveProtocol,
                 safety: SafetyManagerProtocol,
                 controller: ControllerProtocol,
                 logger: Optional[LoggerProtocol] = None,
                 config: PipelineConfig = None):
        """
        Initialize control pipeline.
        
        Args:
            observer: Observation builder
            executive: Decision maker
            safety: Safety enforcer
            controller: Torque computer
            logger: Optional logger
            config: Pipeline configuration
        """
        self.observer = observer
        self.executive = executive
        self.safety = safety
        self.controller = controller
        self.logger = logger
        self.config = config or PipelineConfig()
        
        self.t = 0.0
        self.step_count = 0
        self.episode_active = False
    
    def step(self, model: mj.MjModel, data: mj.MjData, dt: float) -> tuple:
        """
        Execute one control step.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            dt: Timestep
            
        Returns:
            (obs, decision, tau) tuple
        """
        # 1. Build observation
        obs = self.observer.observe(model, data, self.t, dt)
        
        # 2. Generate decision
        decision = self.executive.decide(obs)
        
        # 3. Enforce safety
        decision_safe = self.safety.enforce(decision, obs)
        
        # 4. Compute torques
        # For MPC-based controllers we must supply a joint target; otherwise the
        # controller will default to holding position.  Use the shared
        # decision_to_joint_target helper to convert the executive's end‑
        # effector target into a joint‑space target.  Non‑MPC controllers
        # (e.g. PD, impedance) ignore the additional argument.
        if self.config and self.config.controller_type in ("joint_mpc", "task_mpc"):
            # Lazy import to avoid circular dependency
            from .targets import decision_to_joint_target
            q_target = decision_to_joint_target(model, data, obs, decision_safe)
            tau = self.controller.compute_torques(model, data, obs, decision_safe, q_target=q_target)
        else:
            tau = self.controller.compute_torques(model, data, obs, decision_safe)
        
        # 5. Log
        if self.logger:
            self.logger.log_step(obs, decision_safe, tau)
        
        # Update time
        self.t += dt
        self.step_count += 1
        
        return obs, decision_safe, tau
    
    def start_episode(self):
        """Start new episode."""
        self.t = 0.0
        self.step_count = 0
        self.episode_active = True
        self.executive.reset()
        
        if hasattr(self.controller, 'reset'):
            self.controller.reset()
    
    def end_episode(self, success: bool = False, failure_reason: str = None):
        """End current episode."""
        self.episode_active = False
        
        if self.logger:
            self.logger.log_event("episode_end", {
                "success": success,
                "failure_reason": failure_reason,
                "steps": self.step_count,
                "duration": self.t
            })


def create_pipeline(model: mj.MjModel, config: PipelineConfig = None) -> ControlPipeline:
    """
    Factory function to create a complete control pipeline.
    
    Args:
        model: MuJoCo model
        config: Pipeline configuration
        
    Returns:
        Configured ControlPipeline instance
    """
    config = config or PipelineConfig()

    # Normalize controller type using canonical mapping.  This ensures aliases
    # such as "task_space_mpc" map to the canonical "task_mpc".  Update
    # config.controller_type in place so downstream logic sees the canonical
    # value.
    from .controller_types import normalize_controller_type
    try:
        config.controller_type = normalize_controller_type(config.controller_type)
    except Exception as e:
        raise
    
    # Create observer
    observer = MujocoObserver(model, config.task_name)
    
    # Create profile library and state machine
    from .profiles_v2 import ProfileLibraryV2
    from .state_machine import RFSNStateMachine
    from .profiles import ProfileLibrary  # Legacy for state machine
    
    profile_library_v2 = ProfileLibraryV2()
    legacy_profile_library = ProfileLibrary()
    state_machine = RFSNStateMachine(config.task_name, legacy_profile_library)
    
    # Create learner if enabled
    learner = None
    if config.enable_learning:
        from .learner_v2 import ContextualProfileLearner
        learner = ContextualProfileLearner(
            state_names=profile_library_v2.list_states(),
            variants=["base", "precise", "smooth", "fast", "stable"],
            dim=20  # Context dimension
        )
    
    # Create executive
    executive = RFSNExecutive(state_machine, learner, profile_library_v2)
    
    # Create safety manager
    safety = SafetyManagerV2()
    
    # Create controller
    controller = ControllerFactory.create(
        config.controller_type, model,
        {'planning_interval': config.planning_interval}
    )
    
    # Create logger if enabled
    logger = None
    if config.enable_logging:
        from .logger import RFSNLogger
        logger = RFSNLogger()
    
    return ControlPipeline(observer, executive, safety, controller, logger, config)
