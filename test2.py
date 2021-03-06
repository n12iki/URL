import networkx as pt
import matplotlib.pyplot as pst
import operator
def k_centers_prob(V, n):
  centers = []
  cities = V.nodes()
  centers.append((V.nodes())[0])
  cities.remove(centers[0]) 
  n = n-1 
  while n!= 0:
    city_dict = {}
    for cty in cities:
      min_dist = float("inf")
      for c in centers:
        min_dist = min(min_dist,V[cty][c]['length'])
      city_dict[cty] = min_dist
    new_center = max(city_dict, key = lambda i: city_dict[i])
    centers.append(new_center)
    cities.remove(new_center)
    n = n-1
  return centers
def cGraph():
  V = pt.Graph()
  f = open('input.txt')
  n = int(f.readline()) 
  wtMatrix = []
  for i in range(n):
    list1 = map(int, (f.readline()).split())
    wtMatrix.append(list1)
  for i in range(n) :
    for j in range(n)[i:] :
        V.add_edge(i, j, length = wtMatrix[i][j]) 
  noc = int(f.readline()) 
  return V, noc
def dGraph(V, centers):
  pos = pt.spring_layout(V)
   color_map = ['blue'] * len(V.nodes())
   for c in centers:
     color_map[c] = 'red'
  pt.draw(V, pos, node_color = color_map, with_labels = True) 
  edge_labels = pt.get_edge_attributes(V, 'length')
  pt.draw_networkx_edge_labels(V, pos, edge_labels = edge_labels, font_size = 11) 
  
#main function
if __name__ == "__main__":
  V,n = cGraph()
  c = k_centers_prob(V, n)
  dGraph(V, centers)
  pst.show()