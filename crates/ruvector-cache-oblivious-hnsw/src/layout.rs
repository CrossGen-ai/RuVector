//! Permutation generators for the three physical layouts.
//!
//! Each returns `perm[logical_id] = slot`, a `Vec<u32>` of length `n`.
//! They differ in how neighbouring logical nodes get packed in memory.

use std::collections::VecDeque;

/// Adjacency helper: `adj[i]` is the list of logical neighbour ids of node `i`.
pub type Adj = Vec<Vec<u32>>;

/// Breadth-first-search layout — the natural baseline. Node visitation order
/// starting from `entry` becomes the physical slot order. Frontiers land in
/// consecutive slots; nodes reachable in the same ply cluster.
pub fn bfs_permutation(n: usize, entry: u32, adj: &Adj) -> Vec<u32> {
    let mut perm = vec![u32::MAX; n];
    let mut next_slot: u32 = 0;
    let mut queue: VecDeque<u32> = VecDeque::new();
    queue.push_back(entry);
    perm[entry as usize] = next_slot;
    next_slot += 1;
    while let Some(u) = queue.pop_front() {
        for &v in &adj[u as usize] {
            if perm[v as usize] == u32::MAX {
                perm[v as usize] = next_slot;
                next_slot += 1;
                queue.push_back(v);
            }
        }
    }
    // Any disconnected nodes get appended in id order.
    for id in 0..n as u32 {
        if perm[id as usize] == u32::MAX {
            perm[id as usize] = next_slot;
            next_slot += 1;
        }
    }
    perm
}

/// Depth-first-search layout. A greedy path from `entry` gets packed as a
/// contiguous run of slots, then the next path, and so on. Descent-heavy
/// searches following a single beam benefit from this.
pub fn dfs_permutation(n: usize, entry: u32, adj: &Adj) -> Vec<u32> {
    let mut perm = vec![u32::MAX; n];
    let mut next_slot: u32 = 0;
    let mut stack: Vec<u32> = Vec::with_capacity(n);
    stack.push(entry);
    while let Some(u) = stack.pop() {
        if perm[u as usize] != u32::MAX {
            continue;
        }
        perm[u as usize] = next_slot;
        next_slot += 1;
        // Push neighbours in reverse so the first unvisited neighbour is
        // popped next → contiguous descent path.
        for &v in adj[u as usize].iter().rev() {
            if perm[v as usize] == u32::MAX {
                stack.push(v);
            }
        }
    }
    for id in 0..n as u32 {
        if perm[id as usize] == u32::MAX {
            perm[id as usize] = next_slot;
            next_slot += 1;
        }
    }
    perm
}

/// Van Emde Boas layout over the BFS tree spanning the graph from `entry`.
///
/// Method: build the BFS tree; recursively split it at half its depth into a
/// top subtree and a set of bottom subtrees, laying each block out
/// contiguously. Nodes whose ancestor distance is < `sqrt(depth)` land near
/// each other in memory, which is the classical cache-oblivious guarantee:
/// every fixed cache line size B gets an amortized O(log_B n) memory-transfer
/// bound on root-to-leaf traversal.
pub fn veb_permutation(n: usize, entry: u32, adj: &Adj) -> Vec<u32> {
    // Build BFS tree: parent[v] = u for the edge that first discovered v.
    let mut parent: Vec<i32> = vec![-1; n];
    let mut depth: Vec<u32> = vec![0; n];
    let mut order: Vec<u32> = Vec::with_capacity(n);
    let mut queue: VecDeque<u32> = VecDeque::new();
    let mut seen = vec![false; n];
    queue.push_back(entry);
    seen[entry as usize] = true;
    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &v in &adj[u as usize] {
            if !seen[v as usize] {
                seen[v as usize] = true;
                parent[v as usize] = u as i32;
                depth[v as usize] = depth[u as usize] + 1;
                queue.push_back(v);
            }
        }
    }

    // Group children by parent for tree walk.
    let mut children: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &v in order.iter() {
        let p = parent[v as usize];
        if p >= 0 {
            children[p as usize].push(v);
        }
    }

    let max_depth = *depth.iter().max().unwrap_or(&0);

    // Recursive vEB layout on the BFS tree. Each call lays out the subtree
    // rooted at `root` whose depth range is [d_lo, d_hi], writing slots
    // starting at `*next_slot`.
    fn layout_veb(
        root: u32,
        d_lo: u32,
        d_hi: u32,
        children: &[Vec<u32>],
        depth: &[u32],
        perm: &mut [u32],
        next_slot: &mut u32,
    ) {
        if d_lo == d_hi {
            if perm[root as usize] == u32::MAX {
                perm[root as usize] = *next_slot;
                *next_slot += 1;
            }
            return;
        }
        let d_mid = d_lo + (d_hi - d_lo) / 2;
        // Collect roots of bottom subtrees: descendants at depth d_mid+1.
        let mut bottom_roots: Vec<u32> = Vec::new();
        // First, lay out the top block (nodes at depth in [d_lo, d_mid]).
        // We do this with a small stack DFS collecting frontier nodes.
        let mut stack: Vec<u32> = vec![root];
        while let Some(u) = stack.pop() {
            if perm[u as usize] == u32::MAX {
                perm[u as usize] = *next_slot;
                *next_slot += 1;
            }
            for &c in &children[u as usize] {
                if depth[c as usize] <= d_mid {
                    stack.push(c);
                } else if depth[c as usize] == d_mid + 1 {
                    bottom_roots.push(c);
                }
            }
        }
        // Recursively lay out each bottom subtree contiguously.
        for br in bottom_roots {
            layout_veb(br, d_mid + 1, d_hi, children, depth, perm, next_slot);
        }
    }

    let mut perm = vec![u32::MAX; n];
    let mut next_slot: u32 = 0;
    layout_veb(entry, 0, max_depth, &children, &depth, &mut perm, &mut next_slot);
    // Disconnected → id order.
    for id in 0..n as u32 {
        if perm[id as usize] == u32::MAX {
            perm[id as usize] = next_slot;
            next_slot += 1;
        }
    }
    perm
}
