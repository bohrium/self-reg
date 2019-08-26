''' author: samuel tenka
    change: 2019-06-16 
    create: 2019-06-16 
    descrp: get all SGD diagrams   
'''

from itertools import permutations as perm

def get_all_partitions(nb_nodes):
    if nb_nodes==0: return [[]]
    old_partitions = get_all_partitions(nb_nodes-1)
    new_partitions = []
    new_node = nb_nodes-1
    for pp in old_partitions:
        new_partitions.append(pp + [(new_node,)]) # singleton
        for i in range(len(pp)): # nonsingleton
            new_partitions.append([p if j!=i else p+(new_node,) for j,p in enumerate(pp)])
    return new_partitions

def get_all_trees(nb_nodes):
    if nb_nodes <= 1: return [[]]
    old_trees = get_all_trees(nb_nodes-1)
    new_trees = []
    for gg in old_trees:
        ggg = [(s+1, e+1) for (s,e) in gg] 
        for j in range(1, nb_nodes): 
            new_trees.append([(0,j)] + ggg)
    return new_trees

def name_of_diagram(partition, tree):
    return '(%s)(%s)' % (
        '-'.join(''.join(str(s) for s in g) for g in tree),
        '-'.join(''.join(str(s) for s in p) for p in partition)
    )

def canonical_form(nb_nodes, partition, tree):
    return min(
        (
            tuple(sorted(tuple(sorted(p[el] for el in edge)) for edge in tree)),
            tuple(sorted(tuple(sorted(p[el] for el in part)) for part in partition))
        )
        for p in perm(range(nb_nodes))
    )
    
def get_diagrams(nb_nodes): 
    rtrn = set()
    for tree in get_all_trees(nb_nodes):
        for partition in get_all_partitions(nb_nodes):
            rtrn.add(canonical_form(nb_nodes, partition, tree))
    return sorted(rtrn)

for nb_nodes in range(1, 5):
    for tree, partition in get_diagrams(nb_nodes):
        print(name_of_diagram(partition, tree))
 
