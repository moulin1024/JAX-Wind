"""Run a cryogenic case with disposable state buffers to reduce GPU memory.

Use on a compute node. The state passed to advancement must not be reused;
checkpointing and diagnostics are provided by the ordinary runtime.
"""
from __future__ import annotations

import argparse
from dataclasses import replace


def with_donated_state(simulation):
    import jax
    from jaxwind.simulation.api import RunControls

    original_advance = simulation.advance_block

    @jax.jit(static_argnums=(1,), donate_argnums=(0,))
    def advance(state, count, target_time):
        return original_advance(state, RunControls(count, target_time))

    def advance_unique(state, controls):
        # Empty parcel fields can share a zero buffer, including between blocks.
        # Donation requires each input buffer to be owned by exactly one leaf.
        seen = set()

        def unique(array):
            pointer = array.unsafe_buffer_pointer()
            if pointer in seen:
                return array.copy()
            seen.add(pointer)
            return array

        state = jax.tree.map(unique, state)
        return advance(state, controls.count, controls.target_time)

    return replace(simulation, advance_block=advance_unique)


def main():
    from jaxwind.config.document import load_case
    from jaxwind.simulation.api import build_simulation
    from jaxwind.runtime.engine import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    case = load_case(args.case)
    if not case.formulation.startswith("cryogenic-"):
        parser.error("this runner is for cryogenic cases")
    simulation = with_donated_state(build_simulation(case))
    result = run(case, output=args.output, max_steps=args.max_steps, _simulation=simulation)
    print(result.summary, flush=True)


if __name__ == "__main__":
    main()
