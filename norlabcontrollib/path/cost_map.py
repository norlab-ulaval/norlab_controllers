import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class ImgCostMap:
    def __init__(self, img_path, square_size, x_scale, y_scale):
        self.square_size = square_size
        
        # Grid creation
        self.x_grid = np.arange(x_scale[0], x_scale[1] + square_size[0], square_size[0])
        self.y_grid = np.arange(y_scale[0], y_scale[1] + square_size[1], square_size[1])
        self.grid = np.zeros((len(self.y_grid), len(self.x_grid)))

        # Image reading
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_height, img_width = img.shape

        x_indices = np.linspace(0, img_width-1, self.grid.shape[1], dtype=int)
        y_indices = np.linspace(0, img_height-1, self.grid.shape[0], dtype=int)

        self.grid = 255 - img[np.ix_(y_indices, x_indices)]
        self.graph = self.create_grid_graph()

    def get_indexes(self, coords):
        x, y = coords

        x_idx = np.argmin(np.abs(self.x_grid - x))
        y_idx = np.argmin(np.abs(self.y_grid - y))

        # Invert y index to match image coordinates
        return (len(self.grid[0])-1) - y_idx, x_idx

    def get_coords(self, indexes):
        i, j = indexes

        return self.x_grid[j], self.y_grid[(len(self.grid[0])-1) - i]

    def __getitem__(self, coords):
        i, j = self.get_indexes(coords)
        
        return self.grid[i, j]

    def create_grid_graph(self):
        G = nx.Graph()
        rows, cols = self.grid.shape
        
        for i in range(rows):
            for j in range(cols):
                G.add_node((i, j), cost=0)

                G.add_edge((i, j), (i, j), weight=0)
                
                if i > 0:
                    G.add_edge((i, j), (i-1, j), weight=self.grid[i, j])
                if i < rows - 1:
                    G.add_edge((i, j), (i+1, j), weight=self.grid[i, j])
                if j > 0:
                    G.add_edge((i, j), (i, j-1), weight=self.grid[i, j])
                if j < cols - 1:
                    G.add_edge((i, j), (i, j+1), weight=self.grid[i, j])
        
        return G

    def plot(self):
        extent = [
            self.x_grid[0] - self.square_size[0] / 2,
            self.x_grid[-1] + self.square_size[0] / 2,
            self.y_grid[0] - self.square_size[1] / 2,
            self.y_grid[-1] + self.square_size[1] / 2,
        ]

        plt.imshow(self.grid, extent=extent, cmap='gray')

        plt.vlines(self.x_grid + self.square_size[0] / 2, extent[2], extent[3], colors='gray', linestyles='-', linewidth=0.25)
        plt.hlines(self.y_grid + self.square_size[1] / 2, extent[0], extent[1], colors='gray', linestyles='-', linewidth=0.25)
