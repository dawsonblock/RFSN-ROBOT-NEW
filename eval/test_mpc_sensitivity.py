"""
V10 Acceptance Test: MPC Sensitivity Validation
===============================================
Tests that MPC parameters have measurable impact on behavior.

Acceptance Criteria:
1. Same seed with two different configs produces different behavior
2. Config A (conservative): small horizon, high R, high du → smoother, slower
3. Config B (aggressive): large horizon, low R, low du → faster, more aggressive
4. Solve time remains within budget most steps
5. MPC failure rate is low
"""

import numpy as np
import mujoco as mj
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rfsn.mpc_receding import RecedingHorizonMPCQP, MPCConfig, OSQP_AVAILABLE


# Test configuration constants
CONVERGENCE_POSITION_THRESHOLD = 0.15  # rad - position error for convergence
CONVERGENCE_VELOCITY_THRESHOLD = 0.2   # rad/s - velocity threshold for convergence


def run_mpc_trajectory(config_name, mpc_config, decision_params, seed=42):
    """
    Run a simple trajectory with given MPC config.
    
    Returns metrics: total_time, smoothness, energy, solve_times, failures
    """
    np.random.seed(seed)
    
    # Load model
    model = mj.MjModel.from_xml_path("panda_table_cube.xml")
    data = mj.MjData(model)
    
    # Initialize
    mj.mj_resetData(model, data)
    data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    mj.mj_forward(model, data)
    
    # Target position (move to different configuration - closer target)
    q_target = np.array([0.2, -0.6, 0.1, -2.2, 0.1, 1.6, 0.9])
    
    # Joint limits
    q_min = model.jnt_range[:7, 0]
    q_max = model.jnt_range[:7, 1]
    
    # Create MPC solver
    solver = RecedingHorizonMPCQP(mpc_config)
    
    # Run trajectory
    dt = 0.002
    max_steps = 500
    
    metrics = {
        'solve_times': [],
        'costs': [],
        'q_errors': [],
        'qd_norms': [],
        'qdd_norms': [],
        'qdd_changes': [],
        'failures': 0,
        'total_steps': 0
    }
    
    q = data.qpos[:7].copy()
    qd = data.qvel[:7].copy()
    qdd_prev = np.zeros(7)
    
    start_time = time.perf_counter()
    
    for step in range(max_steps):
        # Solve MPC
        result = solver.solve(
            q=q,
            qd=qd,
            q_target=q_target,
            dt=dt,
            decision_params=decision_params,
            joint_limits=(q_min, q_max)
        )
        
        # Record metrics
        metrics['solve_times'].append(result.solve_time_ms)
        metrics['costs'].append(result.cost_total)
        metrics['q_errors'].append(np.linalg.norm(q - q_target))
        metrics['qd_norms'].append(np.linalg.norm(qd))
        
        if not result.converged:
            metrics['failures'] += 1
        
        if result.qdd_cmd_next is not None:
            qdd = result.qdd_cmd_next
            metrics['qdd_norms'].append(np.linalg.norm(qdd))
            metrics['qdd_changes'].append(np.linalg.norm(qdd - qdd_prev))
            qdd_prev = qdd
        else:
            qdd = np.zeros(7)
            metrics['qdd_norms'].append(0.0)
            metrics['qdd_changes'].append(0.0)
        
        # Use MPC output to update state
        if result.q_ref_next is not None and result.qd_ref_next is not None:
            q = result.q_ref_next
            qd = result.qd_ref_next
        else:
            # Fallback: integrate with zero acceleration
            qd = qd + dt * qdd
            q = q + dt * qd
        
        metrics['total_steps'] += 1
        
        # Check convergence
        if np.linalg.norm(q - q_target) < CONVERGENCE_POSITION_THRESHOLD and np.linalg.norm(qd) < CONVERGENCE_VELOCITY_THRESHOLD:
            break
    
    # Compute aggregate metrics (total_time not used, removed)
    avg_solve_time = np.mean(metrics['solve_times'])
    max_solve_time = np.max(metrics['solve_times'])
    smoothness = np.mean(metrics['qdd_changes'])  # Lower is smoother
    energy = np.sum(metrics['qdd_norms'])  # Total effort
    time_to_goal = metrics['total_steps'] * dt
    failure_rate = metrics['failures'] / metrics['total_steps'] if metrics['total_steps'] > 0 else 1.0
    final_error = metrics['q_errors'][-1] if metrics['q_errors'] else float('inf')
    initial_error = metrics['q_errors'][0] if metrics['q_errors'] else float('inf')
    
    print(f"\n{config_name} Results:")
    print(f"  Time to goal: {time_to_goal:.3f} s ({metrics['total_steps']} steps)")
    print(f"  Initial error: {initial_error:.4f} rad")
    print(f"  Final error: {final_error:.4f} rad")
    print(f"  Error reduction: {(initial_error - final_error):.4f} rad")
    print(f"  Smoothness (avg Δqdd): {smoothness:.4f}")
    print(f"  Energy (total effort): {energy:.2f}")
    print(f"  Avg solve time: {avg_solve_time:.2f} ms")
    print(f"  Max solve time: {max_solve_time:.2f} ms")
    print(f"  Failure rate: {failure_rate*100:.1f}%")
    
    return {
        'time_to_goal': time_to_goal,
        'final_error': final_error,
        'q_errors': metrics['q_errors'],
        'smoothness': smoothness,
        'energy': energy,
        'avg_solve_time': avg_solve_time,
        'max_solve_time': max_solve_time,
        'failure_rate': failure_rate,
        'total_steps': metrics['total_steps'],
        'failures': metrics['failures']
    }


def test_mpc_sensitivity():
    """
    Test that MPC parameters have measurable impact.
    
    Config A: Conservative (small horizon, high R, high du)
    Config B: Aggressive (large horizon, low R, low du)
    """
    print("=" * 70)
    print("V10 MPC SENSITIVITY TEST")
    print("=" * 70)
    
    if not OSQP_AVAILABLE:
        print("\n✗ SKIP: OSQP not available, cannot test QP MPC")
        return False
    
    print("\nTesting that MPC parameters produce measurably different behavior...")
    print()
    
    # Config A: Conservative
    print("=" * 70)
    print("CONFIG A: Conservative (small horizon, high R, high du)")
    print("=" * 70)
    
    mpc_config_a = MPCConfig(
        H_min=5,
        H_max=30,
        max_iterations=100,
        time_budget_ms=50.0,
        warm_start=True
    )
    
    decision_params_a = {
        'horizon_steps': 8,  # Small horizon
        'Q_diag': np.concatenate([
            50.0 * np.ones(7),  # Position tracking
            10.0 * np.ones(7)   # Velocity penalty
        ]),
        'R_diag': 0.05 * np.ones(7),  # High effort penalty (conservative)
        'terminal_Q_diag': np.concatenate([
            100.0 * np.ones(7),
            10.0 * np.ones(7)
        ]),
        'du_penalty': 0.05  # High smoothness penalty
    }
    
    metrics_a = run_mpc_trajectory("Config A", mpc_config_a, decision_params_a, seed=42)
    
    # Config B: Aggressive
    print("\n" + "=" * 70)
    print("CONFIG B: Aggressive (large horizon, low R, low du)")
    print("=" * 70)
    
    mpc_config_b = MPCConfig(
        H_min=5,
        H_max=30,
        max_iterations=100,
        time_budget_ms=50.0,
        warm_start=True
    )
    
    decision_params_b = {
        'horizon_steps': 25,  # Large horizon
        'Q_diag': np.concatenate([
            50.0 * np.ones(7),  # Same position tracking
            10.0 * np.ones(7)   # Same velocity penalty
        ]),
        'R_diag': 0.01 * np.ones(7),  # Low effort penalty (aggressive)
        'terminal_Q_diag': np.concatenate([
            100.0 * np.ones(7),
            10.0 * np.ones(7)
        ]),
        'du_penalty': 0.01  # Low smoothness penalty
    }
    
    metrics_b = run_mpc_trajectory("Config B", mpc_config_b, decision_params_b, seed=42)
    
    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    print(f"\nTime to goal:")
    print(f"  Config A: {metrics_a['time_to_goal']:.3f} s")
    print(f"  Config B: {metrics_b['time_to_goal']:.3f} s")
    print(f"  Difference: {abs(metrics_a['time_to_goal'] - metrics_b['time_to_goal']):.3f} s")
    
    print(f"\nSmoothness (avg Δqdd):")
    print(f"  Config A: {metrics_a['smoothness']:.4f}")
    print(f"  Config B: {metrics_b['smoothness']:.4f}")
    print(f"  Difference: {abs(metrics_a['smoothness'] - metrics_b['smoothness']):.4f}")
    
    print(f"\nEnergy (total effort):")
    print(f"  Config A: {metrics_a['energy']:.2f}")
    print(f"  Config B: {metrics_b['energy']:.2f}")
    print(f"  Difference: {abs(metrics_a['energy'] - metrics_b['energy']):.2f}")
    
    print(f"\nSolve time:")
    print(f"  Config A avg: {metrics_a['avg_solve_time']:.2f} ms")
    print(f"  Config B avg: {metrics_b['avg_solve_time']:.2f} ms")
    print(f"  Config A max: {metrics_a['max_solve_time']:.2f} ms")
    print(f"  Config B max: {metrics_b['max_solve_time']:.2f} ms")
    
    # Acceptance criteria (simple, measurable)
    criteria_met = []

    # 1) Configs produce measurably different behavior
    energy_diff_pct = abs(metrics_a['energy'] - metrics_b['energy']) / max(metrics_a['energy'], metrics_b['energy'], 1e-9) * 100.0
    smoothness_diff_pct = abs(metrics_a['smoothness'] - metrics_b['smoothness']) / max(metrics_a['smoothness'], metrics_b['smoothness'], 1e-9) * 100.0
    criterion_1 = (energy_diff_pct > 20.0) or (smoothness_diff_pct > 20.0) or (abs(metrics_a['time_to_goal'] - metrics_b['time_to_goal']) > 0.05)
    criteria_met.append(("Different behavior (energy/smooth/time)", criterion_1))

    # 2) Conservative config should be smoother-or-equal
    criterion_2 = metrics_a['smoothness'] <= metrics_b['smoothness']
    criteria_met.append(("Conservative smoother-or-equal", criterion_2))

    # 3) Solve time is not exploding
    criterion_3 = (metrics_a['max_solve_time'] < 100.0) and (metrics_b['max_solve_time'] < 100.0)
    criteria_met.append(("Solve time under 100ms max", criterion_3))

    # 4) Both configs make progress
    criterion_4 = (metrics_a['q_errors'][0] - metrics_a['final_error']) > 0.0 and (metrics_b['q_errors'][0] - metrics_b['final_error']) > 0.0
    criteria_met.append(("Both reduce error", criterion_4))

    # 5) Failure rate isn't pathological
    criterion_5 = (metrics_a['failure_rate'] < 0.5) and (metrics_b['failure_rate'] < 0.5)
    criteria_met.append(("Failure rate < 50%", criterion_5))
    
    all_passed = all(passed for _, passed in criteria_met)

    # One-line verdict (what you actually want to see in CI logs)
    verdict = "PASS" if all_passed else "FAIL"
    print(
        f"MPC_SENSITIVITY: {verdict} "
        f"tA={metrics_a['time_to_goal']:.3f}s tB={metrics_b['time_to_goal']:.3f}s "
        f"smoothA={metrics_a['smoothness']:.4f} smoothB={metrics_b['smoothness']:.4f} "
        f"E_A={metrics_a['energy']:.2f} E_B={metrics_b['energy']:.2f} "
        f"solveA(avg/max)={metrics_a['avg_solve_time']:.2f}/{metrics_a['max_solve_time']:.2f}ms "
        f"solveB(avg/max)={metrics_b['avg_solve_time']:.2f}/{metrics_b['max_solve_time']:.2f}ms "
        f"failA={metrics_a['failure_rate']*100:.1f}% failB={metrics_b['failure_rate']*100:.1f}%"
    )

    # Keep the human-readable criteria if you run locally
    for label, passed in criteria_met:
        status = "✓" if passed else "✗"
        print(f"  {status} {label}")

    return all_passed


def main():
    """Run MPC sensitivity test."""
    print("\n" + "=" * 70)
    print("V10 MPC SENSITIVITY ACCEPTANCE TEST")
    print("=" * 70)
    print()
    
    try:
        passed = test_mpc_sensitivity()
        return 0 if passed else 1
    except Exception as e:
        print(f"\n✗ TEST CRASHED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
