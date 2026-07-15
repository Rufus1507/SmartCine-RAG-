import networkx as nx
from collections import deque

def clean_name(node_id: str) -> str:
    """
    Loại bỏ tiền tố của ID node để lấy tên hiển thị thực tế.
    Ví dụ: 'Movie:Inception' -> 'Inception'
    """
    if ":" in node_id:
        return node_id.split(":", 1)[1]
    return node_id

def explain_path_from_nodes(graph: nx.MultiDiGraph, path: list[str]) -> str:
    """
    Giải thích đường đi quan hệ đã được tìm thấy từ trước mà không cần chạy lại shortest_path.
    
    Args:
        graph: Đồ thị phim NetworkX MultiDiGraph.
        path: Danh sách các node tạo thành đường đi.
        
    Returns:
        str: Câu giải thích tiếng Việt.
    """
    path_len = len(path)
    if path_len < 2:
        return ""
        
    node_a = path[0]
    node_b = path[-1]
    name_a = clean_name(node_a)
    name_b = clean_name(node_b)
    
    if path_len == 3:
        # movie_a -> node_1 -> movie_b
        node_1 = path[1]
        name_1 = clean_name(node_1)
        t1 = graph.nodes[node_1].get("type", "Unknown")
        if t1 == "Genre":
            return f"Cả hai phim '{name_a}' và '{name_b}' đều thuộc thể loại {name_1}."
        elif t1 == "Country":
            return f"Cả hai phim '{name_a}' và '{name_b}' đều được sản xuất tại {name_1}."
        elif t1 == "Director":
            return f"Đạo diễn {name_1} đã chỉ đạo cả hai phim '{name_a}' và '{name_b}'."
        elif t1 == "Actor":
            return f"Diễn viên {name_1} đều góp mặt trong cả hai phim '{name_a}' và '{name_b}'."
            
    elif path_len == 4:
        # movie_a -> node_1 -> node_2 -> movie_b
        node_1 = path[1]
        node_2 = path[2]
        name_1 = clean_name(node_1)
        name_2 = clean_name(node_2)
        t1 = graph.nodes[node_1].get("type", "Unknown")
        t2 = graph.nodes[node_2].get("type", "Unknown")
        
        if t1 == "Director" and t2 == "Actor":
            return f"Đạo diễn {name_1} của phim '{name_a}' đã từng hợp tác với diễn viên {name_2}, người đóng trong phim '{name_b}'."
        elif t1 == "Actor" and t2 == "Director":
            return f"Diễn viên {name_1} trong phim '{name_a}' đã từng đóng phim của đạo diễn {name_2}, người chỉ đạo phim '{name_b}'."
        elif t1 == "Actor" and t2 == "Actor":
            return f"Diễn viên {name_1} trong phim '{name_a}' đã từng hợp tác với diễn viên {name_2}, người đóng trong phim '{name_b}'."
        elif t1 == "Director" and t2 == "Director":
            return f"Đạo diễn {name_1} của phim '{name_a}' đã từng hợp tác với đạo diễn {name_2} của phim '{name_b}'."
            
    elif path_len == 5:
        # movie_a -> node_1 -> movie_2 -> node_3 -> movie_b
        node_1 = path[1]
        movie_2 = path[2]
        node_3 = path[3]
        name_1 = clean_name(node_1)
        name_2 = clean_name(movie_2)
        name_3 = clean_name(node_3)
        t1 = graph.nodes[node_1].get("type", "Unknown")
        t3 = graph.nodes[node_3].get("type", "Unknown")
        
        rel_1 = "có sự tham gia của" if t1 == "Actor" else "được đạo diễn bởi"
        rel_3 = "có sự tham gia của" if t3 == "Actor" else "được đạo diễn bởi"
        
        return f"Phim '{name_a}' {rel_1} {name_1}, người cũng liên kết qua phim trung gian '{name_2}', bộ phim mà {name_3} ({rel_3} {name_3}) cũng tham gia để kết nối tới phim '{name_b}'."
        
    # Trường hợp đường đi dài hơn hoặc tổng quát
    explanation_parts = []
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        utype = graph.nodes[u].get("type")
        vtype = graph.nodes[v].get("type")
        u_clean = clean_name(u)
        v_clean = clean_name(v)
        
        if utype == "Movie" and vtype == "Director":
            explanation_parts.append(f"được đạo diễn bởi {v_clean}")
        elif utype == "Movie" and vtype == "Actor":
            explanation_parts.append(f"có diễn viên {v_clean}")
        elif utype == "Movie" and vtype == "Genre":
            explanation_parts.append(f"thuộc thể loại {v_clean}")
        elif utype == "Movie" and vtype == "Country":
            explanation_parts.append(f"sản xuất tại {v_clean}")
        elif utype == "Director" and vtype == "Movie":
            explanation_parts.append(f"chỉ đạo phim '{v_clean}'")
        elif utype == "Actor" and vtype == "Movie":
            explanation_parts.append(f"đóng trong phim '{v_clean}'")
        elif utype == "Director" and vtype == "Actor":
            explanation_parts.append(f"hợp tác với diễn viên {v_clean}")
        elif utype == "Actor" and vtype == "Director":
            explanation_parts.append(f"hợp tác với đạo diễn {v_clean}")
        elif utype == "Actor" and vtype == "Actor":
            explanation_parts.append(f"đồng đóng phim với {v_clean}")
            
    return f"Phim '{name_a}' " + " -> ".join(explanation_parts) + f" để kết nối tới '{name_b}'."

def explain_path(graph: nx.MultiDiGraph, movie_a: str, movie_b: str) -> tuple[str, str]:
    """
    Trả về tuple (câu giải thích tiếng Việt, loại liên kết 'personnel' | 'shared_attribute')
    về mối quan hệ giữa 2 bộ phim trong đồ thị.
    """
    # Chuẩn hóa tên phim trong đồ thị
    def find_node_case_insensitive(name):
        if name.startswith("Movie:") and graph.has_node(name):
            return name
        prefixed = f"Movie:{name}"
        if graph.has_node(prefixed):
            return prefixed
        name_lower = name.lower()
        for node, data in graph.nodes(data=True):
            if data.get("type") == "Movie" and clean_name(node).lower() == name_lower:
                return node
        return None

    node_a = find_node_case_insensitive(movie_a)
    node_b = find_node_case_insensitive(movie_b)
    
    if not node_a or not node_b:
        return f"Không tìm thấy thông tin liên kết giữa phim '{movie_a}' và '{movie_b}'.", "shared_attribute"
        
    if node_a == node_b:
        return "Hai phim là một.", "personnel"
        
    # Tạo đồ thị vô hướng ảo để tìm đường đi ngắn nhất
    U = graph.to_undirected(as_view=True)
    try:
        # Lấy tất cả các đường đi ngắn nhất để lọc ưu tiên
        shortest_paths = list(nx.all_shortest_paths(U, source=node_a, target=node_b))
        
        # Ưu tiên tìm đường đi chỉ đi qua nhân sự (personnel)
        selected_path = shortest_paths[0]
        for p in shortest_paths:
            is_personnel = True
            for node in p[1:-1]:
                ntype = graph.nodes[node].get("type", "Unknown")
                if ntype in ("Genre", "Country"):
                    is_personnel = False
                    break
            if is_personnel:
                selected_path = p
                break
                
        # Phân loại đường đi đã chọn
        p_type = "personnel"
        for node in selected_path[1:-1]:
            ntype = graph.nodes[node].get("type", "Unknown")
            if ntype in ("Genre", "Country"):
                p_type = "shared_attribute"
                break
                
        explanation = explain_path_from_nodes(graph, selected_path)
        return explanation, p_type
    except nx.NetworkXNoPath:
        return f"Không có liên kết trực tiếp giữa phim '{clean_name(node_a)}' và phim '{clean_name(node_b)}' trên đồ thị.", "shared_attribute"

def get_limited_neighbors(graph: nx.MultiDiGraph, u: str, max_neighbors_per_hop: int = 20, personnel_only: bool = False) -> list[tuple[str, str]]:
    """
    Lấy danh sách các node hàng xóm liên kết với node u, có giới hạn số lượng và sắp xếp ưu tiên.
    """
    raw_connections = []
    if graph.has_node(u):
        for v in graph.successors(u):
            for key in graph[u][v]:
                etype = graph[u][v][key].get("type")
                if personnel_only and etype not in ("DIRECTED", "ACTED_IN", "COLLAB_WITH"):
                    continue
                weight = graph[u][v][key].get("weight", 1)
                raw_connections.append((v, etype, weight))
        for v in graph.predecessors(u):
            for key in graph[v][u]:
                etype = graph[v][u][key].get("type")
                if personnel_only and etype not in ("DIRECTED", "ACTED_IN", "COLLAB_WITH"):
                    continue
                weight = graph[v][u][key].get("weight", 1)
                raw_connections.append((v, etype, weight))
                
    by_node = {}
    for v, etype, weight in raw_connections:
        vtype = graph.nodes[v].get("type")
        if v not in by_node:
            by_node[v] = (etype, weight, vtype)
            
    movies = []
    others = []
    
    for v, (etype, weight, vtype) in by_node.items():
        if vtype == "Movie":
            rating = graph.nodes[v].get("rating", 0.0) or 0.0
            votes = graph.nodes[v].get("num_votes", 0) or 0
            movies.append((v, etype, weight, rating, votes))
        else:
            others.append((v, etype, weight))
            
    movies = sorted(movies, key=lambda x: (x[3], x[4]), reverse=True)
    if len(movies) > max_neighbors_per_hop:
        movies = movies[:max_neighbors_per_hop]
        
    others = sorted(others, key=lambda x: x[2], reverse=True)
    if len(others) > max_neighbors_per_hop:
        others = others[:max_neighbors_per_hop]
        
    result = []
    for m in movies:
        result.append((m[0], m[1]))
    for o in others:
        result.append((o[0], o[1]))
        
    return result

def find_collaborators_of_movie(graph: nx.MultiDiGraph, movie_title: str) -> list[dict]:
    """
    Trả về tất cả Đạo diễn/Diễn viên đã từng hợp tác với cast/crew của bộ phim movie_title.
    """
    actual_movie_node = None
    if movie_title.startswith("Movie:") and graph.has_node(movie_title):
        actual_movie_node = movie_title
    elif graph.has_node(f"Movie:{movie_title}"):
        actual_movie_node = f"Movie:{movie_title}"
    else:
        movie_lower = movie_title.lower()
        for node, data in graph.nodes(data=True):
            if data.get("type") == "Movie" and clean_name(node).lower() == movie_lower:
                actual_movie_node = node
                break
                
    if not actual_movie_node:
        return []
        
    cast_and_crew = []
    for u in graph.predecessors(actual_movie_node):
        ntype = graph.nodes[u].get("type")
        if ntype in ("Director", "Actor"):
            cast_and_crew.append(u)
            
    collaborators = {}
    for person in cast_and_crew:
        if graph.has_node(person):
            for neighbor in graph.neighbors(person):
                if graph.has_edge(person, neighbor, key="COLLAB_WITH"):
                    edge_data = graph[person][neighbor]["COLLAB_WITH"]
                    weight = edge_data.get("weight", 0)
                    ntype = graph.nodes[neighbor].get("type")
                    
                    if neighbor not in cast_and_crew and neighbor != actual_movie_node:
                        if neighbor not in collaborators:
                            collaborators[neighbor] = {
                                "name": clean_name(neighbor),
                                "type": ntype,
                                "collaborator_of": clean_name(person),
                                "weight": weight
                            }
                        else:
                            collaborators[neighbor]["weight"] += weight
                            
    return sorted(collaborators.values(), key=lambda x: x["weight"], reverse=True)

def find_movies_by_collab_path(
    graph: nx.MultiDiGraph, 
    reference_movie_title: str, 
    max_hops: int = 3, 
    max_neighbors_per_hop: int = 20
) -> list[dict]:
    """
    Từ một phim tham chiếu, đi qua Actor/Director/Genre/Country, tìm các phim khác liên quan qua đường đi đồ thị.
    """
    actual_movie_node = None
    if reference_movie_title.startswith("Movie:") and graph.has_node(reference_movie_title):
        actual_movie_node = reference_movie_title
    elif graph.has_node(f"Movie:{reference_movie_title}"):
        actual_movie_node = f"Movie:{reference_movie_title}"
    else:
        movie_lower = reference_movie_title.lower()
        for node, data in graph.nodes(data=True):
            if data.get("type") == "Movie" and clean_name(node).lower() == movie_lower:
                actual_movie_node = node
                break
                
    if not actual_movie_node:
        return []
        
    # Hàm hỗ trợ chạy BFS tìm kiếm
    def run_bfs(personnel_only: bool) -> dict[str, list[str]]:
        queue = deque([(actual_movie_node, [actual_movie_node])])
        visited = {actual_movie_node}
        found_movies = {}
        
        while queue:
            curr, path = queue.popleft()
            hop_count = len(path) - 1
            
            if hop_count >= max_hops:
                continue
                
            neighbors = get_limited_neighbors(graph, curr, max_neighbors_per_hop, personnel_only=personnel_only)
            for neighbor, etype in neighbors:
                if neighbor in visited:
                    continue
                    
                new_path = path + [neighbor]
                visited.add(neighbor)
                
                vtype = graph.nodes[neighbor].get("type")
                if vtype == "Movie":
                    found_movies[neighbor] = new_path
                    
                queue.append((neighbor, new_path))
        return found_movies

    # 1. Thử nghiệm tìm kiếm bằng personnel path trước (chỉ đi qua Actor/Director/Collab)
    candidate_movies = run_bfs(personnel_only=True)
    
    # 2. Nếu số lượng phim tìm được quá ít (< 5), fallback chạy BFS đầy đủ với cả Genre/Country
    if len(candidate_movies) < 5:
        candidate_movies = run_bfs(personnel_only=False)
        
    result = []
    for m_node, path in candidate_movies.items():
        # Tối ưu hóa: Sinh câu giải thích trực tiếp từ path tìm được mà không cần tính lại shortest_path
        explanation = explain_path_from_nodes(graph, path)
        
        # Phân loại đường đi
        p_type = "personnel"
        for node in path[1:-1]:
            ntype = graph.nodes[node].get("type", "Unknown")
            if ntype in ("Genre", "Country"):
                p_type = "shared_attribute"
                break
                
        node_data = graph.nodes[m_node]
        
        result.append({
            "Title": clean_name(m_node),
            "Rating": node_data.get("rating"),
            "Year": node_data.get("year"),
            "num_votes": node_data.get("num_votes"),
            "genres": node_data.get("genres"),
            "decade": node_data.get("decade"),
            "has_awards": node_data.get("has_awards"),
            "has_oscar": node_data.get("has_oscar"),
            "has_nomination": node_data.get("has_nomination"),
            "hop_count": len(path) - 1,  # Số bước từ phim gốc tới phim ứng viên (1 = cùng diễn viên/đạo diễn trực tiếp)
            "graph_path_explanation": explanation,
            "graph_path_type": p_type
        })
        
    return result

def find_top_collaborator(graph: nx.MultiDiGraph, person_name: str, top_k: int = 5) -> list[dict]:
    """
    Tim top collaborator cua mot Director/Actor dua tren trong so canh COLLAB_WITH.
    person_name: ten director hoac actor (vi du "Christopher Nolan").

    P4 FIX: Graph co canh bidirectional (d->a va a->d deu co COLLAB_WITH weight=N).
    Phien ban cu cong weight tu ca successors lan predecessors -> bi dem 2 lan.
    Fix: chi duyet mot chieu (successors), neu khong co thi moi duyet predecessors
    (truong hop person la Actor - duoc Director->Actor COLLAB_WITH point to).
    """
    person_node = None
    person_name_lower = person_name.lower()
    for node, data in graph.nodes(data=True):
        if data.get("type") in ("Director", "Actor") and clean_name(node).lower() == person_name_lower:
            person_node = node
            break
    
    if not person_node:
        return []
    
    collaborators = {}
    if graph.has_node(person_node):
        # Chi duyet successors (chieu d_node -> a_node) de tranh double-count.
        # Do graph co a_node -> d_node la canh nguoc cung COLLAB_WITH weight,
        # neu dem ca predecessors se bi nhan doi so lan hop tac.
        for v in graph.successors(person_node):
            for key in graph[person_node][v]:
                etype = graph[person_node][v][key].get("type")
                if etype == "COLLAB_WITH":
                    weight = graph[person_node][v][key].get("weight", 1)
                    vtype = graph.nodes[v].get("type")
                    vname = clean_name(v)
                    if v not in collaborators:
                        collaborators[v] = {"name": vname, "type": vtype, "weight": weight}
                    else:
                        collaborators[v]["weight"] += weight
        
        # Neu khong co successors COLLAB_WITH (person la Actor, khong phai Director),
        # thi moi duyet predecessors de lay Director -> Actor direction.
        if not collaborators:
            for v in graph.predecessors(person_node):
                for key in graph[v][person_node]:
                    etype = graph[v][person_node][key].get("type")
                    if etype == "COLLAB_WITH":
                        weight = graph[v][person_node][key].get("weight", 1)
                        vtype = graph.nodes[v].get("type")
                        vname = clean_name(v)
                        if v not in collaborators:
                            collaborators[v] = {"name": vname, "type": vtype, "weight": weight}
                        else:
                            collaborators[v]["weight"] += weight
    
    result = sorted(collaborators.values(), key=lambda x: x["weight"], reverse=True)
    return result[:top_k]
