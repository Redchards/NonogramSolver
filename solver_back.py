#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from enum import Enum
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import *
import time
import math



class GridPresenter:
    def __init__(self, width, height, step_count = 25):
        self.width = width
        self.height = height
        self.step_count = step_count
        self.img = Image.new(mode='L', size=(height, width), color=255)
        self.step_size = int(self.width / self.step_count)

        self.print_grid()
        
        
    def print_grid(self):
        printer = ImageDraw.Draw(self.img)

        y_start = 0
        y_end = self.height
    
        for x in range(0, self.width, self.step_size):
            line = ((x, y_start), (x, y_end))
            printer.line(line, fill=128)
    
        x_start = 0
        x_end = self.width
    
        for y in range(0, self.height, self.step_size):
            line = ((x_start, y), (x_end, y))
            printer.line(line, fill=128)
    
        del printer
                
    def set_color(self, i, j, c):
        printer = ImageDraw.Draw(self.img)
        
        printer.rectangle([(i * self.step_size, j * self.step_size), ((i + 1) * self.step_size, (j + 1) * self.step_size)], fill=c)
            
        del printer
        #self.img.show()
        
    def show(self):
        self.img.show()
        
    def load_grid_data(self, grid_data):
        print(len(grid_data), len(grid_data[-1]))
        for i in range(len(grid_data)):
            for j in range(len(grid_data[0])):
                if grid_data[i][j] == DynamicSolver.Color.BLACK:
                    self.set_color(i, j, "#000000")

class DynamicSolver:
    class For(Enum):
        LINE = 0,
        COLUMN = 1
        
    class Color(Enum):
        UNDEFINED = 0,
        WHITE = 1,
        BLACK = 2
    
    def __init__(self):
        self.line_validity_trace = []
        self.column_validity_trace = []
        self.unload_instance()
        
    def unload_instance(self):
        self.instance = [[]]
        self.lSeq = []
        self.cSeq = []
        
    def load_instance(self, filename):
        f = open(filename, "r")
        
        isLineSeq = True
        for line in f:
            if line[:-1] == "#":
                isLineSeq = False
                continue
            if line[-1] == '\n':
                seq = [int(x) for x in line[:-1].split(" ") if x != ""]
            else:
                seq = [int(x) for x in line.split(" ") if x != ""]
            if isLineSeq:
                self.lSeq.append(seq)
            else:
                self.cSeq.append(seq)
                
        #self.line_validity_trace = [[None] * len(self.cSeq) for i in range(len(self.lSeq))]
        #self.column_validity_trace = [[None] * len(self.cSeq) for i in range(len(self.lSeq))]
        self.current_line = -1
        self.current_column = -1
        
        self.instance = [[self.Color.UNDEFINED] * len(self.cSeq) for i in range(len(self.lSeq))]
                
        #print(self.instance)
        f.close()
        
    def get_instance_as_JSON(self):
        return "{\"hor\":" + str(self.cSeq) + ",\"ver\":" + str(self.lSeq) + "}"
        
    def compute_validity_trace(self, i, j, l, t=For.LINE):
        if t == self.For.LINE:
            #print("Hi ! ",i, j, l)
            if self.current_line != i:
                self.current_line = i
                self.line_validity_trace = [[None] * len(self.cSeq) for i in range(len(self.lSeq[i]) + 1)]
                #print(self.line_validity_trace)
                #print(self.lSeq[i])
            if self.line_validity_trace[l][j] is None:
                self.line_validity_trace[l][j] = self.compute_validity_trace_line(i, j, l)
            #print(self.line_validity_trace)
 
            return self.line_validity_trace[l][j]
        else:
            #print("Hi ! ",i, j, l)
            #print(self.column_validity_trace)

            if self.current_column != j:
                self.current_column = j
                self.column_validity_trace = [[None] * len(self.lSeq) for i in range(len(self.cSeq[j]) + 1)]
                #print(self.cSeq[j])
                #print(self.column_validity_trace)
            if self.column_validity_trace[l][i] is None:
                self.column_validity_trace[l][i] = self.compute_validity_trace_column(i, j, l)
            #print(self.column_validity_trace[l])
            return self.column_validity_trace[l][i]
    
    # TODO : si l == 0, alors pas de cases noires
    def compute_validity_trace_line(self, i, j, l):
        if l == 0:
            if j == 0:
                #print("here")
                return self.instance[i][j] != self.Color.BLACK 
            return self.instance[i][j] != self.Color.BLACK and self.compute_validity_trace(i, j - 1, 0)
        
        sl = self.lSeq[i][l - 1]
        #print(sl, j, l)
        if j < sl - 1:
            return False
        if self.instance[i][j] == self.Color.WHITE:
            #print("(1)")
            #print(j)
            if j - 1 >= 0:
                return self.compute_validity_trace(i, j - 1, l, self.For.LINE)
            else:
                return False
        else:
            #print("check")
            if self.instance[i][j] == self.Color.UNDEFINED and self.compute_validity_trace(i, j - 1, l, self.For.LINE):
                #print("(3.1)")
                return True
            found_white_cell = False
            for j_prime in range(j - sl + 1, j + 1):
                if self.instance[i][j_prime] == self.Color.WHITE:
                    found_white_cell = True
                    break
            if found_white_cell:
                #print("(3.2)")
                return False


        if j - sl - 1 >= 0:
            if self.instance[i][j - sl] == self.Color.BLACK:
                return False
            return self.compute_validity_trace(i, j - sl - 1, l - 1, self.For.LINE)
        elif l - 1 == 0:
            return True
        else:
            return False
        
    def compute_validity_trace_column(self, i, j, l):
        if l == 0:
            if i == 0:
               return self.instance[i][j] != self.Color.BLACK 
            return self.instance[i][j] != self.Color.BLACK and self.compute_validity_trace(i - 1, j, 0, self.For.COLUMN)
        
        #print(self.cSeq[j])
        #print(l - 1)
        #print(i, self.cSeq[j])

        sl = self.cSeq[j][l - 1]
        if i < sl - 1:
            return False
        if self.instance[i][j] == self.Color.WHITE:
            #print("(1)")
            #print(j)
            if i - 1 >= 0:
                return self.compute_validity_trace(i - 1, j, l, self.For.COLUMN)
            else:
                return False
        else:
            #print("check")
            if self.instance[i][j] == self.Color.UNDEFINED and self.compute_validity_trace(i - 1, j, l, self.For.COLUMN):
                return True
            found_white_cell = False
            for i_prime in range(i - sl + 1, i + 1):
                if self.instance[i_prime][j] == self.Color.WHITE:
                    found_white_cell = True
                    break
            if found_white_cell:
                #print("(3.2)")
                return False

        #print(i - sl - 1, sl, i)
        if i - sl >= 0:
            if self.instance[i - sl][j] == self.Color.BLACK:
                return False
        if i - sl - 1 >= 0:
            #print("here")
            return self.compute_validity_trace(i - sl - 1, j, l - 1, self.For.COLUMN)
        elif l - 1 == 0:
            return True
        else:
            return False
        
    def set_cell_color(self, i, j, c):
        # TODO  :test if C is of type enum Color
        self.instance[i][j] = c
        
    def show_grid(self):
        grid = np.array([[0x000000 if j == DynamicSolver.Color.BLACK else 0xffffff if j != DynamicSolver.Color.UNDEFINED else 0xaaaaaa for j in self.instance[i]] for i in range(len(self.instance))])

        plt.imshow(grid, cmap="gray")
        #plt.gca().tick_params(axis="x", direction="out", top=1, bottom=0, labelbottom=0, labeltop=1)
        #plt.xticks(range(grid.shape[1]), [ str(x)[1:-1].replace(',','\n') for x in self.cSeq], rotation='horizontal')
        #plt.yticks(range(grid.shape[0]), [ str(x)[1:-1].replace(',',' ') for x in self.lSeq])
        
    def solve(self):
        instance_height = len(self.instance)
        instance_width = len(self.instance[0])
        instance_size = instance_height * instance_width
        fixed_cells_count = 0
        
        line_to_explore = [i for i in range(instance_height)]
        column_to_explore = [j for j in range(instance_width)]
        
        while fixed_cells_count != instance_size:
            
            if len(line_to_explore) == 0 and len(column_to_explore) == 0:
                return "ERROR"

            for i in line_to_explore:
                for j in range(instance_width):
                    #print(line_to_explore)
                    #print(i, j)
            
                    if self.instance[i][j] == self.Color.UNDEFINED:
        
                        #print("testing line : ", i, ":", j, " with ", self.lSeq[i])
                        self.set_cell_color(i, j, self.Color.BLACK)
                        # UGLY HACK !!
                        self.line_validity_trace = [[None] * len(self.cSeq) for i in range(len(self.lSeq[i]) + 1)]
                        
                        success_black_coloring = self.compute_validity_trace(i, instance_width - 1, len(self.lSeq[i]), self.For.LINE)
                        #print("Success black coloring ? ", success_black_coloring)
                        #success_black_coloring = (line_test and column_test)
                        self.set_cell_color(i, j, self.Color.WHITE)
                        
                        # UGLY HACK !!
                        self.line_validity_trace = [[None] * len(self.cSeq) for i in range(len(self.lSeq[i]) + 1)]
                        
                        success_white_coloring = self.compute_validity_trace(i, instance_width - 1, len(self.lSeq[i]), self.For.LINE)
                        #print("Success white coloring ?", success_white_coloring)
                        #column_test = self.compute_validity_trace(i, j, len(self.cSeq[j]), self.For.COLUMN)
                            
                        #success_white_coloring = (line_test and column_test)
                        had_change = False
                        if not success_black_coloring and success_white_coloring:    
                            self.set_cell_color(i, j, self.Color.WHITE)
                            fixed_cells_count += 1
                            had_change = True
                            #print("color white")
                        elif success_black_coloring and not success_white_coloring:
                            self.set_cell_color(i, j, self.Color.BLACK)
                            fixed_cells_count += 1
                            had_change = True
                            #print("color black")
                        elif not success_black_coloring and not success_white_coloring:
                            return "ERROR"
                        else:
                            self.set_cell_color(i, j, self.Color.UNDEFINED)
                            #print("no coloring")
                            
                        if had_change and not j in column_to_explore:
                            column_to_explore.append(j)
                            
            line_to_explore = []

            for j in column_to_explore:
                for i in range(instance_height):
                    #print(self.instance)

                    if self.instance[i][j] == self.Color.UNDEFINED:
                        #print(self.cSeq, j)
                        #print("testing column : ", i, ":", j, " with ", self.cSeq[j])
                        self.set_cell_color(i, j, self.Color.BLACK)
                        # UGLY HACK !!
                        self.column_validity_trace = [[None] * len(self.lSeq) for i in range(len(self.cSeq[j]) + 1)]
                        
                        success_black_coloring = self.compute_validity_trace(instance_height - 1, j, len(self.cSeq[j]), self.For.COLUMN)
                        #print("Success black coloring ? ", success_black_coloring)
                        #success_black_coloring = (line_test and column_test)
                        self.set_cell_color(i, j, self.Color.WHITE)
                        
                        # UGLY HACK !!
                        self.column_validity_trace = [[None] * len(self.lSeq) for i in range(len(self.cSeq[j]) + 1)]
                        
                        success_white_coloring = self.compute_validity_trace(instance_height - 1, j, len(self.cSeq[j]), self.For.COLUMN)
                        #print("Success white coloring ?", success_white_coloring)
                        #column_test = self.compute_validity_trace(i, j, len(self.cSeq[j]), self.For.COLUMN)
                            
                        #success_white_coloring = (line_test and column_test)
                        had_change = False
                        
                        if not success_black_coloring and success_white_coloring:    
                            self.set_cell_color(i, j, self.Color.WHITE)
                            fixed_cells_count += 1
                            had_change = True
                            #print("color white")
                        elif success_black_coloring and not success_white_coloring:
                            self.set_cell_color(i, j, self.Color.BLACK)
                            fixed_cells_count += 1
                            had_change = True
                            #print("color black")
                        elif not success_black_coloring and not success_white_coloring:
                            return "ERROR"
                        else:
                            self.set_cell_color(i, j, self.Color.UNDEFINED)
                            #print("no coloring")  
                            
                        if had_change and not i in line_to_explore:
                            line_to_explore.append(i)
                            
            column_to_explore = []
            
class LinearSolver:
    class Color(Enum):
        UNDEFINED = 0,
        WHITE = 1,
        BLACK = 2
    
    def __init__(self):
        self.variable_count = 0
        self.instance_width = 0
        self.instance_height = 0
        self.lSeq = []
        self.cSeq = []
        
    def load_instance(self, filename):
        f = open(filename, "r")
        self.variable_count = 0
        self.instance_file = filename
        
        isLineSeq = True
        for line in f:
            if line[:-1] == "#":
                isLineSeq = False
                continue
            if line[-1] == '\n':
                seq = [int(x) for x in line[:-1].split(" ") if x != ""]
            else:
                seq = [int(x) for x in line.split(" ") if x != ""]
            if isLineSeq:
                self.lSeq.append(seq)
            else:
                self.cSeq.append(seq)
                
        
        self.instance = [[self.Color.UNDEFINED] * len(self.cSeq) for i in range(len(self.lSeq))]
        self.instance_width = len(self.instance[0])
        self.instance_height = len(self.instance)
        self.variable_count += self.instance_width * self.instance_height
        
        for seq in self.lSeq:
            self.variable_count += len(seq) * self.instance_width
        for seq in self.cSeq:
            self.variable_count += len(seq) * self.instance_height
            
        self.model = Model(filename)     

                
        #print(self.instance)
        f.close()
        
    def build_model_variables(self, partial_solution):
        x = []
        y = []
        z = []
        
        for i in range(self.instance_height):
            for j in range(self.instance_width):
                if not partial_solution is None and partial_solution[i][j] == DynamicSolver.Color.BLACK:
                    x.append(self.model.addVar(vtype=GRB.BINARY, lb=1, ub=1, name="x%d,%d" % (((i), (j)))))
                elif not partial_solution is None and partial_solution[i][j] == DynamicSolver.Color.WHITE:
                    x.append(self.model.addVar(vtype=GRB.BINARY, lb=0, ub=0, name="x%d,%d" % (((i), (j)))))
                else:
                    x.append(self.model.addVar(vtype=GRB.BINARY, name="x%d,%d" % (((i), (j)))))

            
        for i in range(self.instance_height):
            l = i * self.instance_width 
            for j in range(self.instance_width):
                y.append([])
                acc = 0
                seq_sum = sum(self.lSeq[i])
                num_seq = len(self.lSeq[i])
                
                for t in range(len(self.lSeq[i])):
                    st = self.lSeq[i][t]
                    #print("Alpha : ", i, j, t, acc + t)
                    #print("Beta : ", i, j, t, self.instance_width - (seq_sum - acc +  num_seq - t - 1))
                    if (j < acc + t) or (j > self.instance_width - (seq_sum - acc +  num_seq - t - 1)):
                        y[l + j].append(None)
                        acc += st
                        continue
                    else:
                        #print("add var")
                        y[l + j].append(self.model.addVar(vtype=GRB.BINARY, name="y%d,%d,%d" % ((i), (j), (t + 1))))
                    acc += st
                
        for j in range(self.instance_width):
            c = j * self.instance_height
            for i in range(self.instance_height):
                z.append([])
                acc = 0
                seq_sum = sum(self.cSeq[j])
                num_seq = len(self.cSeq[j])
                
                for t in range(len(self.cSeq[j])):
                    st = self.cSeq[j][t]
                    if (i < acc + t) or i > (self.instance_height - (seq_sum - acc  + num_seq - t - 1)):
                        z[c + i].append(None)
                        acc += st
                        continue
                    else:
                        z[c + i].append(self.model.addVar(vtype=GRB.BINARY, name="z%d,%d,%d" % ((i), (j), (t + 1))))
                    acc += st
        return x, y, z
    
    def show_grid(self):
        grid = np.array([[0x000000 if j == 1 else 0xffffff for j in self.instance[i]] for i in range(len(self.instance))])

        plt.imshow(grid, cmap="gray")
        #plt.gca().tick_params(axis="x", direction="out", top=1, bottom=0, labelbottom=0, labeltop=1)
        #plt.xticks(range(grid.shape[1]), [ str(x)[1:-1].replace(',','\n') for x in self.cSeq], rotation='horizontal')
        #plt.yticks(range(grid.shape[0]), [ str(x)[1:-1].replace(',',' ') for x in self.lSeq])
            
        
    def solve(self, use_presolving = False, seed = 35594):
        partial_solution = None
        if use_presolving:
            presolver = DynamicSolver()
            presolver.load_instance(self.instance_file)
            presolver.solve()
            partial_solution = presolver.instance
            
        x, y, z = self.build_model_variables(partial_solution)
        
        self.model.update()

        
        for i in range(self.instance_height):
            l = i * self.instance_width 
            self.model.addConstr(quicksum(x[l + k] for k in range(self.instance_width)) == sum(self.lSeq[i]))

            max_t = 0
            for j in range(self.instance_width):
                for t in range(len(y[l + j])):
                    max_t = max(t, max_t)
                    seq_var = y[l + j][t]
                    if seq_var is None:
                        continue
                    st = self.lSeq[i][t]
                    self.model.addConstr(quicksum(x[l + k] for k in range(j, j + st)) >= st * seq_var)
                    if t < len(y[l + j]) - 1:
                        #print(j)
                        #if j + st + 1 < self.instance_width:
                        self.model.addConstr(quicksum(y[l + k][t + 1] for k in range(j + st + 1, self.instance_width) if not y[l + k][t + 1] is None) >= seq_var)
                        #else:
                        #    self.model.addConstr(y[j + st][t + 1] < seq_var, "Constraint (2.1 bis) : %d" % i)
            if not len(y[l]) == 0:
                for t in range(max_t + 1):
                    self.model.addConstr(quicksum(y[l + k][t] for k in range(0, self.instance_width) if not y[l + k][t] is None) == 1)
                        
        for j in range(self.instance_width):
            c = j * self.instance_height
            max_t = 0
            self.model.addConstr(quicksum(x[j + k * self.instance_width] for k in range(self.instance_height)) == sum(self.cSeq[j]))

            for i in range(self.instance_height):
                for t in range(len(z[c + i])):
                    max_t = max(t, max_t)
                    seq_var = z[c + i][t]
                    if seq_var is None:
                        continue
                    st = self.cSeq[j][t]
                    self.model.addConstr(quicksum(x[j + k * self.instance_width] for k in range(i, i + st)) >= st * seq_var)
                    if t < len(z[c + i]) - 1:
                        self.model.addConstr(quicksum(z[c + k][t + 1] for k in range(i + st + 1, self.instance_height) if not z[c + k][t + 1] is None) >= seq_var)
            if not len(z[c]) == 0:
                for t in range(max_t + 1):  
                    self.model.addConstr(quicksum(z[c + k][t] for k in range(0, self.instance_height) if not z[c + k][t] is None) == 1)

        self.model.setObjective(quicksum(x[idx] for idx in range(len(x))) ,GRB.MAXIMIZE)
        #self.model.update()
        self.model.setParam(GRB.Param.Seed, seed)
        #self.model.setParam(GRB.Param.TimeLimit)
        self.model.write("debug.lp")

        self.model.optimize()
        self.instance = [[x[idx].x for idx in range(l * self.instance_width, (l + 1) * self.instance_width)] for l in range(self.instance_height)]
        
            
                    

"""
solver = DynamicSolver()
solver.load_instance("instances/11.txt")
#solver.set_cell_color(0, 4, DynamicSolver.Color.BLACK)
#solver.set_cell_color(0, 0, DynamicSolver.Color.BLACK)
#solver.set_cell_color(2, 2, DynamicSolver.Color.WHITE)
#solver.set_cell_color(2, 1, DynamicSolver.Color.WHITE)
#solver.set_cell_color(2, 0, DynamicSolver.Color.BLACK)
print(solver.instance)
#print(solver.compute_validity_trace(1, 3, 1, DynamicSolver.For.COLUMN))
print(solver.cSeq)
solver.solve()


#solver.set_cell_color(2, 3, DynamicSolver.Color.BLACK)
#print(solver.compute_validity_trace(2, 4, 3, DynamicSolver.For.LINE))
print(solver.instance)

#grid = GridPresenter(1000, 1000)
#grid.load_grid_data(solver.instance)
#grid.show()
plt.imshow(np.array([[0x000000 if j == DynamicSolver.Color.BLACK else 0xffffff for j in solver.instance[i]] for i in range(len(solver.instance))]), cmap='gray')
print(solver.get_instance_as_JSON())"""
presolver = LinearSolver()
presolver.load_instance("instances/9.txt")
presolver.solve(True, 21488320)

"""best = -1
best_time = math.inf
for i in range(100):

    s = int(np.random.uniform(low=0, high=100000000))
    print("testing seed : ", s)
    presolver = LinearSolver()
    presolver.load_instance("instances/15.txt")
    start = time.time()
    presolver.solve(True, s)
    end = time.time()
    if end - start < best_time:
        best = s
        best_time = end - start
        print(best_time, best)  

        

print(best_time, best)""" 

presolver.show_grid()
#21488320
#