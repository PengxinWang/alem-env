"""CPU-only masked-random rollout with a GIF, per-step trace and JSON summary."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import numpy as np
from PIL import Image, ImageDraw

from algorithms.random.agent import (
    build_env, sample_random_actions, stack_agents, unstack_actions,
    team_achievement_names,
)
from alem.alem_coop.constants import (
    Action, BLOCK_PIXEL_SIZE_IMG, TEXTURES, load_player_specific_textures,
)
from alem.alem_coop.renderer.renderer_pixels import render_alem_pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coord", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--output", type=Path, default=Path("outputs/random_demo"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = build_env({
        "ENV_NAME": "Alem-Coop-Symbolic", "MAX_TIMESTEPS": 10000,
        "SHARED_REWARD": False, "NUM_AGENTS": 3,
        "TRAINING_COORDINATION_DIFFICULTY": args.coord, "COMPUTE_FULL_INFO": False,
    })
    agents = tuple(env.agents)
    rng, reset_rng = jax.random.split(jax.random.PRNGKey(args.seed))
    print("Generating world / compiling reset on CPU...", flush=True)
    obs, state = jax.block_until_ready(env.reset(reset_rng))
    obs_shapes = {agent: list(value.shape) for agent, value in obs.items()}
    action_dim = env.action_space(agents[0]).n
    comm_base = len(Action) + max(0, env.num_agents - 2)

    def action_name(value):
        value = int(value)
        if value >= comm_base:
            return f"COMM_CHANNEL_{value - comm_base}"
        if value >= Action.GIVE.value:
            return f"GIVE_TEAMMATE_SLOT_{value - Action.GIVE.value}"
        return Action(value).name

    textures = load_player_specific_textures(TEXTURES[BLOCK_PIXEL_SIZE_IMG], 3)

    @jax.jit
    def render(state):
        return render_alem_pixels(state, BLOCK_PIXEL_SIZE_IMG, env.static_env_params, textures)

    @jax.jit
    def advance(state, rng):
        rng, action_rng, step_rng = jax.random.split(rng, 3)
        masks = stack_agents(env.get_avail_actions(state), agents)
        actions = sample_random_actions(action_rng, masks, action_dim, True)
        _, next_state, rewards, dones, _ = env.step_env(
            step_rng, state, unstack_actions(actions, agents)
        )
        return next_state, rng, actions, stack_agents(rewards, agents), dones["__all__"]

    frames = []

    def capture(step):
        pixels = np.asarray(render(state)).clip(0, 255).astype(np.uint8)
        height, width = pixels.shape[1:3]
        frame = Image.new("RGB", (3 * width, height + 42), "#17202b")
        draw = ImageDraw.Draw(frame)
        for i in range(3):
            frame.paste(Image.fromarray(pixels[i]), (i * width, 42))
            draw.text((i * width + 5, 5), f"Agent {i} | HP {float(state.player_health[i]):.1f}", fill="white")
        draw.text((5, 24), f"Masked random | {args.coord} | seed {args.seed} | step {step}", fill="white")
        frames.append(frame)

    print("Compiling pixel renderer...", flush=True)
    capture(0)
    frames[0].save(args.output / "initial.png")
    returns = np.zeros(3)
    done = False
    print("Compiling environment step / starting random rollout...", flush=True)
    with (args.output / "trace.jsonl").open("w") as trace:
        for step in range(1, args.steps + 1):
            state, rng, actions, rewards, done = jax.block_until_ready(advance(state, rng))
            returns += np.asarray(rewards)
            record = {
                "step": step, "action_ids": np.asarray(actions).tolist(),
                "actions": [action_name(a) for a in actions],
                "rewards": np.asarray(rewards).tolist(),
                "positions_row_col": np.asarray(state.player_position).tolist(),
                "health": np.asarray(state.player_health).tolist(),
                "alive": np.asarray(state.player_alive).tolist(), "done": bool(done),
            }
            trace.write(json.dumps(record) + "\n")
            capture(step)
            if step % 25 == 0 or bool(done) or step == args.steps:
                print(f"step={step} return={returns.sum():.2f} alive={record['alive']}", flush=True)
            if bool(done):
                break
    frames[-1].save(args.output / "final.png")
    frames[0].save(args.output / "replay.gif", save_all=True, append_images=frames[1:],
                   duration=120, loop=0, optimize=False)
    summary = {
        "policy": "uniform random over legal actions", "backend": jax.default_backend(),
        "seed": args.seed, "difficulty": args.coord, "players": 3,
        "requested_steps": args.steps, "completed_steps": step,
        "episode_done": bool(done), "stopped_at_demo_limit": not bool(done),
        "agent_returns": returns.tolist(), "team_return_sum": float(returns.sum()),
        "team_achievements": team_achievement_names(state),
        "alive": np.asarray(state.player_alive).tolist(),
        "observation_shapes": obs_shapes, "action_count": action_dim,
        "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "jax_version": jax.__version__, "elapsed_seconds": round(time.perf_counter() - started, 2),
        "note": "Single short smoke demo; not a leaderboard evaluation.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Replay: {args.output / 'replay.gif'}", flush=True)


if __name__ == "__main__":
    main()
