import numpy as np

nodes = 10
edges = 32

done = False

iter = 0
while not done:
    first = np.random.randint(nodes, size = edges)
    second = np.random.randint(nodes, size = edges)
    
    test = np.zeros(nodes)
    for i in range(nodes):
        test[first[i]] = 1
        test[second[i]] = 1
        
    if np.sum(test) == nodes:
        done = True
        
    if done == False:
        continue
        
    for i in range(edges):
        if first[i] == second[i]:
            if first[i] == 0:
                second[i] = 1
            else:
                second[i] = first[i] - 1
            
    breakLoops = False
    for i in range(edges):
        for j in range(edges):
            if i == j:
                continue
            if (first[i], second[i]) == (first[j], second[j]):
                breakLoops = True
                done = False
                break
        if breakLoops:
            break
        
    iter += 1
    print(iter)
        
graph = []
for i in range(edges):
    graph.append((first[i], second[i]))
    
print(graph)