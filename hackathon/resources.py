from dragon.native.machine import System, Node, cpu_count

if __name__ == '__main__':
    num_cores = cpu_count() // 2   # don't count logical cores
    system = System()
    num_nodes = system.nnodes

    print(f"Hello! Dragon is running on {num_nodes} nodes with a total of {num_cores} CPU cores accessible", flush=True)

    for h_uid in system.nodes:
        node = Node(h_uid)
        print(f"Node {node.hostname} has {len(node.cpus) // 2} CPUs, {node.num_gpus} GPUs, and {node.physical_mem} bytes of memory", flush=True)