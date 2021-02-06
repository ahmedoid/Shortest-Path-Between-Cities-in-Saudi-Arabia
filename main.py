import os
import sys
import time
import traceback
from heapq import heappop, heappush

os.environ["PROJ_LIB"] = os.path.join(os.environ["CONDA_PREFIX"], "share", "proj")

import matplotlib as mpl  # visualization
import pandas as pd  # data manipulate
import networkx as nx  # draw a graph
import matplotlib.pyplot as plt  # show plt
from matplotlib.pyplot import figure, text

from mpl_toolkits.basemap import Basemap as Basemap  # map

# initialize base map to Saudi Arabia
m = Basemap(
    projection='merc',  # map type
    llcrnrlon=32,  # lower corner Longitude
    llcrnrlat=5,  # lower corner latitude
    urcrnrlon=65,  # upper corner Longitude
    urcrnrlat=40,  # upper corner Latitude
    resolution='i',  # intermediate map resolution
)


def get_program_running(start_time):
    end_time = time.time()
    diff_time = end_time - start_time
    result = time.strftime("%H:%M:%S", time.gmtime(diff_time))
    print("program runtime: {}".format(result))


def print_path(graph, weight):
    if weight not in ("length", "time_by_car", "time_by_flight"):
        # so we don't need to check in each branch later
        raise ValueError(f"weight not supported: {weight}")
    print(f"{nx.single_source_dijkstra(graph, 'Taif', weight=weight)}")
    print(f"{nx.single_source_dijkstra_path_length(graph, 'Taif', weight=weight)}")
    path = nx.single_source_dijkstra_path(graph, 'Taif', weight=weight)
    print(f"{path}")


def main():
    try:
        # init graph from networkx package
        graph = nx.Graph()
        # load data file
        df = pd.read_csv('cities.csv')
        # print first few rows
        print(f'Data Head {df.head()}')
        # print data types for all features
        print(f"Data Type {df.dtypes}")
        # split the coronations to lat and long for source, and convert it to float
        df['Long'] = df['source coord'].str.split(',').str[1].astype(float)
        df['Lat'] = df['source coord'].str.split(',').str[0].astype(float)
        # split the coronations to lat and long for target, and convert it to float
        df['LongT'] = df['target coord'].str.split(',').str[1].astype(float)
        df['LatT'] = df['target coord'].str.split(',').str[0].astype(float)
        # add edges to graph with specific wights
        # loop throw all dataframe data

        for index, row in df.iterrows():
            graph.add_edge(row['from'], row['to'], length=row['distance'],
                           time_by_car=row['time by car'], time_by_flight=row['time by flight'])
        # specify color for each edge to make it recognize for legend
        color_edge = {k: v for v, k in enumerate(sorted(set(graph.nodes())))}
        low, *_, high = sorted(color_edge.values())
        norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)

        # create variables to get position for cities to place Edge in correct place base on coordinates
        mx, my = m(df['Long'].values, df['Lat'].values)
        tx, ty = m(df['LongT'].values, df['LatT'].values)
        pos = {}
        # loop throw all cities from and to then add value to pos directory
        for count, elem in enumerate(df['from']):
            pos[elem] = (mx[count], my[count])
        for count, elem in enumerate(df['to']):
            pos[elem] = (tx[count], ty[count])

        # setup plot
        fig, ax = plt.subplots(1, figsize=(12, 12))

        # create labels for Edge
        for label in color_edge:
            ax.plot([0], [0], color=mapper.to_rgba(color_edge[label]), label=label)

        new_labels = dict(
            map(lambda x: ((x[0], x[1]), str(x[2]['length'])), graph.edges(data=True)))
        # add distances between cities
        nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=new_labels, font_size=8)

        # start drawing edge on graph to specify path color, post and edges list
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=graph.edges,
            edge_color='green',
            width=5,
            label="Path",
        )

        # draw the graph with size and position
        nx.draw_networkx(graph, pos,
                         nodelist=color_edge,
                         node_size=2000,
                         node_color=[mapper.to_rgba(i)
                                     for i in color_edge.values()],
                         with_labels=False)
        for node, (x, y) in pos.items():
            # change text size for nodes (City Name)
            text(x, y, node, fontsize=12, ha='center', va='center')
        # add legend
        plt.legend()
        # Now draw the map
        # draw countries border with black color
        m.drawcountries(linewidth=4, color='black')
        # draw coastlines with blue color
        m.drawcoastlines(linewidth=2, color='blue')
        # add title to plot
        plt.title('Shortest Distance between Saudi Arabia Cities by (Distance, Car and Flight)')
        # adjusts subplot params so that the subplot(s) fits in to the figure area
        plt.tight_layout()
        # show graph with map
        plt.show()

        # find short path using differences weight
        # my hone city is Taif near to Makkah, this will be the source for path
        # Shortest path by distance or length
        start_time = time.time()
        print_path(graph, 'length')
        # Shortest path by time by a car
        print_path(graph, 'time_by_car')
        # Shortest path by time by a flight
        print_path(graph, 'time_by_flight')

        get_program_running(start_time)

    except:
        exception_type, exception_value, exception_traceback = sys.exc_info()
        print("Exception Type: {}\nException Value: {}".format(exception_type, exception_value))
        file_name, line_number, procedure_name, line_code = traceback.extract_tb(exception_traceback)[-1]
        print("File Name: {}\nLine Number: {}\nProcedure Name: {}\nLine Code: {}".format(file_name, line_number,
                                                                                         procedure_name, line_code))
    finally:
        pass


if __name__ == '__main__':
    main()
