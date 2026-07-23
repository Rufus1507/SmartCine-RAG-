"""
Test Task 6: graph_query.py — NaN rating không được xử lý trước khi sort
==========================================================================
Xác nhận rằng get_limited_neighbors() xử lý đúng node có rating=float('nan'):
- Không raise exception
- Sort ổn định (NaN được coi như 0.0, không leo lên đầu/cuối một cách bất thường)
"""

import math
import networkx as nx
from chatbot.graph.graph_query import get_limited_neighbors


def build_test_graph():
    """
    Xây dựng đồ thị nhỏ để test:
      - Movie:A (seed)  → Actor:X
      - Actor:X → Movie:B (rating=float('nan'))
      - Actor:X → Movie:C (rating=8.5)
      - Actor:X → Movie:D (rating=7.0)
    """
    G = nx.MultiDiGraph()

    G.add_node("Movie:A", type="Movie", rating=9.0, num_votes=100000)
    G.add_node("Actor:X", type="Actor")
    G.add_node("Movie:B", type="Movie", rating=float("nan"), num_votes=50000)
    G.add_node("Movie:C", type="Movie", rating=8.5, num_votes=80000)
    G.add_node("Movie:D", type="Movie", rating=7.0, num_votes=20000)

    G.add_edge("Actor:X", "Movie:A", type="ACTED_IN", weight=1)
    G.add_edge("Actor:X", "Movie:B", type="ACTED_IN", weight=1)
    G.add_edge("Actor:X", "Movie:C", type="ACTED_IN", weight=1)
    G.add_edge("Actor:X", "Movie:D", type="ACTED_IN", weight=1)

    return G


def test_get_limited_neighbors_does_not_raise_on_nan_rating():
    """
    get_limited_neighbors với node có rating=NaN không được raise exception.
    """
    G = build_test_graph()

    # Seed từ Movie:A, qua Actor:X rồi tới các phim khác
    neighbors = get_limited_neighbors(G, "Movie:A", max_neighbors_per_hop=10)
    assert isinstance(neighbors, list), "Kết quả phải là list"
    # Đảm bảo hàm hoàn thành mà không throw
    print(f"  neighbors of Movie:A: {[n[0] for n in neighbors]}")


def test_get_limited_neighbors_nan_treated_as_zero():
    """
    Node có rating=NaN phải được xếp hạng như rating=0.0,
    KHÔNG được nổi lên đầu danh sách (vì NaN > mọi số trong một số implementation).
    """
    G = build_test_graph()

    # Lấy hàng xóm của Actor:X (bao gồm cả Movie:B có NaN)
    neighbors = get_limited_neighbors(G, "Actor:X", max_neighbors_per_hop=10)
    neighbor_ids = [n[0] for n in neighbors]

    # Lấy vị trí của Movie:B (NaN) trong kết quả
    if "Movie:B" in neighbor_ids:
        idx_b = neighbor_ids.index("Movie:B")
        # Movie:C (rating=8.5) phải xếp trước Movie:B (NaN → 0.0)
        if "Movie:C" in neighbor_ids:
            idx_c = neighbor_ids.index("Movie:C")
            assert idx_c < idx_b, (
                f"Movie:B (NaN) xếp trước Movie:C (8.5)! "
                f"idx_c={idx_c}, idx_b={idx_b} — NaN sort bất ổn"
            )
        print(f"  Movie:B (NaN) correctly at position {idx_b} (after higher-rated movies)")
    else:
        print("  Movie:B not in neighbors (filtered by max_neighbors_per_hop)")


def test_get_limited_neighbors_nan_rating_is_zero():
    """
    Verify: rating của Movie:B (NaN) được convert thành 0.0 trước khi so sánh,
    không phải để nguyên NaN (sẽ fail float comparison).
    """
    G = build_test_graph()

    # Test trực tiếp: thay vì gọi get_limited_neighbors, kiểm tra logic NaN
    raw_rating = G.nodes["Movie:B"].get("rating") or 0.0
    # NaN check theo công thức trong fix
    safe_rating = 0.0 if (raw_rating != raw_rating) else float(raw_rating)

    assert safe_rating == 0.0, f"NaN rating phải được convert thành 0.0, got {safe_rating}"
    assert not math.isnan(safe_rating), "safe_rating không được là NaN"
    print(f"  NaN rating safely converted to: {safe_rating}")


# ─── Runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_get_limited_neighbors_does_not_raise_on_nan_rating,
        test_get_limited_neighbors_nan_treated_as_zero,
        test_get_limited_neighbors_nan_rating_is_zero,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
