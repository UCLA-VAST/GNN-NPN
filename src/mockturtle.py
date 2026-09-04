from __future__ import annotations

import pathlib
import sys

try:
    import mockturtle_wrapper
except ImportError:
    build_python = pathlib.Path(__file__).resolve().parents[1] / "build" / "python"
    sys.path.insert(0, str(build_python))
    import mockturtle_wrapper

from mockturtle_wrapper import aig_to_tt
from mockturtle_wrapper import apply_to_tt
from mockturtle_wrapper import generate_random_aig, generate_random_aig_rewriting
from mockturtle_wrapper import exact_npn, exact_n, exact_p
from mockturtle_wrapper import flip_swap_npn, sifting_npn
from mockturtle_wrapper import cut_aig, rewrite_aig

def get_action_fn(action_name: str | None) -> callable | None:
    if action_name is None:
        return None
    elif action_name == "exact_npn":
        return exact_npn
    elif action_name == "exact_n":
        return exact_n
    elif action_name == "exact_p":
        return exact_p
    elif action_name == "flip_swap_npn":
        return flip_swap_npn
    elif action_name == "sifting_npn":
        return sifting_npn
    else:
        raise ValueError(f"Unknown action: {action_name}")

if __name__ == "__main__":
    # Example usage of mockturtle_wrapper
    print("Generating random AIG...")
    x, e, tt = generate_random_aig(num_pis=3, mean_nodes=5.0, std_nodes=0.1, seed=7)
    print("AIG generated successfully.")
    print("Truth table:\n", tt)
    print("x_data:\n", x)
    print("edge_index:\n", e)

    print("Applying exact NPN canonicalization...")
    tt_opt, in_neg, in_perm, out_neg = exact_npn(tt)
    print("Exact NPN canonicalization applied successfully.")
    print("Optimized truth table:\n", tt_opt)
    print("Input negation:", in_neg)
    print("Input permutation:", in_perm)
    print("Output negation:", out_neg)

    print("Applying transformation")
    tt_transformed = apply_to_tt(tt, in_neg, list(range(len(in_perm))), False)
    print("Input negation applied successfully.")
    print("Transformed truth table:\n", tt_transformed)
    tt_transformed = apply_to_tt(tt_transformed, [False] * len(in_perm), in_perm, False)
    print("Input permutation applied successfully.")
    print("Transformed truth table:\n", tt_transformed)
    tt_transformed = apply_to_tt(tt_transformed, [False] * len(in_perm), list(range(len(in_perm))), out_neg)
    print("Output negation applied successfully.")
    print("Transformed truth table:\n", tt_transformed)

    assert tt_opt == tt_transformed, "The transformed truth table does not match the optimized truth table."
    print("The transformed truth table matches the optimized truth table.")
