#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import *


# The implementation of the solver using the dynamic programming method.
# In fact, it does not use the bottom-up method normally required  by the dynamic
# programming methods, but rather a backward recursion using memoization to avoid
# recomputing the same values again and again.
class DynamicSolver:
    # Used to determine which trace we're computing
    class For(Enum):
        LINE = 0,
        COLUMN = 1
    
    # A simple enumeration to determine the color of the cells
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
    
    # Load an instance from a text file and do the necessary setup
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
                
        self.current_line = -1
        self.current_column = -1
        
        self.instance = [[self.Color.UNDEFINED] * len(self.cSeq) for i in range(len(self.lSeq))]
                
        f.close()
    
    # A function used during the developpement to check the validity of the solution
    # using different online solver, most of which use this particular format.    
    def get_instance_as_JSON(self):
        return "{\"hor\":" + str(self.cSeq) + ",\"ver\":" + str(self.lSeq) + "}"
        
    # Compute the validity trace, the T(j, l).
    def compute_validity_trace(self, i, j, l, t=For.LINE):
        if t == self.For.LINE:
            if self.current_line != i:
                self.current_line = i
                self.line_validity_trace = [[None] * len(self.cSeq) for i in range(len(self.lSeq[i]) + 1)]
            if self.line_validity_trace[l][j] is None:
                self.line_validity_trace[l][j] = self.compute_validity_trace_line(i, j, l)
 
            return self.line_validity_trace[l][j]
        else:
            if self.current_column != j:
                self.current_column = j
                self.column_validity_trace = [[None] * len(self.lSeq) for i in range(len(self.cSeq[j]) + 1)]
            if self.column_validity_trace[l][i] is None:
                self.column_validity_trace[l][i] = self.compute_validity_trace_column(i, j, l)
            
            return self.column_validity_trace[l][i]
    
    # Auxilliary function usd to compute the validity trace for a line
    def compute_validity_trace_line(self, i, j, l):
        if l == 0:
            if j == 0:
                return self.instance[i][j] != self.Color.BLACK 
            return self.instance[i][j] != self.Color.BLACK and self.compute_validity_trace(i, j - 1, 0)
        
        sl = self.lSeq[i][l - 1]
        if j < sl - 1:
            return False
        if self.instance[i][j] == self.Color.WHITE:
            if j - 1 >= 0:
                return self.compute_validity_trace(i, j - 1, l, self.For.LINE)
            else:
                return False
        else:
            if self.instance[i][j] == self.Color.UNDEFINED and self.compute_validity_trace(i, j - 1, l, self.For.LINE):
                return True
            found_white_cell = False
            for j_prime in range(j - sl + 1, j + 1):
                if self.instance[i][j_prime] == self.Color.WHITE:
                    found_white_cell = True
                    break
            if found_white_cell:
                return False


        if j - sl - 1 >= 0:
            if self.instance[i][j - sl] == self.Color.BLACK:
                return False
            return self.compute_validity_trace(i, j - sl - 1, l - 1, self.For.LINE)
        elif l - 1 == 0:
            return True
        else:
            return False
        
    # Auxilliary function usd to compute the validity trace for a line
    def compute_validity_trace_column(self, i, j, l):
        if l == 0:
            if i == 0:
               return self.instance[i][j] != self.Color.BLACK 
            return self.instance[i][j] != self.Color.BLACK and self.compute_validity_trace(i - 1, j, 0, self.For.COLUMN)
        
        sl = self.cSeq[j][l - 1]
        if i < sl - 1:
            return False
        if self.instance[i][j] == self.Color.WHITE:
            if i - 1 >= 0:
                return self.compute_validity_trace(i - 1, j, l, self.For.COLUMN)
            else:
                return False
        else:
            if self.instance[i][j] == self.Color.UNDEFINED and self.compute_validity_trace(i - 1, j, l, self.For.COLUMN):
                return True
            found_white_cell = False
            for i_prime in range(i - sl + 1, i + 1):
                if self.instance[i_prime][j] == self.Color.WHITE:
                    found_white_cell = True
                    break
            if found_white_cell:
                return False

        if i - sl >= 0:
            if self.instance[i - sl][j] == self.Color.BLACK:
                return False
        if i - sl - 1 >= 0:
            return self.compute_validity_trace(i - sl - 1, j, l - 1, self.For.COLUMN)
        elif l - 1 == 0:
            return True
        else:
            return False
        
    def set_cell_color(self, i, j, c):
        # TODO  :test if C is of type enum Color
        self.instance[i][j] = c
    
    # Show the instance grid    
    def show_grid(self):
        grid = np.array([[0x000000 if j == DynamicSolver.Color.BLACK else 0xffffff if j != DynamicSolver.Color.UNDEFINED else 0xaaaaaa for j in self.instance[i]] for i in range(len(self.instance))])

        plt.imshow(grid, cmap="gray")
        #plt.gca().tick_params(axis="x", direction="out", top=1, bottom=0, labelbottom=0, labeltop=1)
        #plt.xticks(range(grid.shape[1]), [ str(x)[1:-1].replace(',','\n') for x in self.cSeq], rotation='horizontal')
        #plt.yticks(range(grid.shape[0]), [ str(x)[1:-1].replace(',',' ') for x in self.lSeq])
    
    # Solve the nonogram using the propagation algorithm given in the subject.    
    def solve(self):
        instance_height = len(self.instance)
        instance_width = len(self.instance[0])
        instance_size = instance_height * instance_width
        fixed_cells_count = 0
        
        # At first, we need to look at every line and column.
        line_to_explore = [i for i in range(instance_height)]
        column_to_explore = [j for j in range(instance_width)]
        
        # If we still have cells to color, we need to continue.
        while fixed_cells_count != instance_size:
            
            # If there is no remaining line or column to explore, but we did not color
            # the full grid, then we failed.
            if len(line_to_explore) == 0 and len(column_to_explore) == 0:
                return "ERROR"
            
            for i in line_to_explore:
                for j in range(instance_width):
            
                    if self.instance[i][j] == self.Color.UNDEFINED:
        
                        self.set_cell_color(i, j, self.Color.BLACK)
                        # Reinit the trace
                        self.line_validity_trace = [[None] * len(self.cSeq) for i in range(len(self.lSeq[i]) + 1)]
                        
                        success_black_coloring = self.compute_validity_trace(i, instance_width - 1, len(self.lSeq[i]), self.For.LINE)
                        
                        #success_black_coloring = (line_test and column_test)
                        self.set_cell_color(i, j, self.Color.WHITE)
                        
                        # Reinit the trace
                        self.line_validity_trace = [[None] * len(self.cSeq) for i in range(len(self.lSeq[i]) + 1)]
                        
                        success_white_coloring = self.compute_validity_trace(i, instance_width - 1, len(self.lSeq[i]), self.For.LINE)
                            
                        #success_white_coloring = (line_test and column_test)
                        had_change = False
                        if not success_black_coloring and success_white_coloring:    
                            self.set_cell_color(i, j, self.Color.WHITE)
                            fixed_cells_count += 1
                            had_change = True
                        elif success_black_coloring and not success_white_coloring:
                            self.set_cell_color(i, j, self.Color.BLACK)
                            fixed_cells_count += 1
                            had_change = True
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
                        self.set_cell_color(i, j, self.Color.BLACK)
                        # Reinit the trace
                        self.column_validity_trace = [[None] * len(self.lSeq) for i in range(len(self.cSeq[j]) + 1)]
                        
                        success_black_coloring = self.compute_validity_trace(instance_height - 1, j, len(self.cSeq[j]), self.For.COLUMN)
                        
                        #success_black_coloring = (line_test and column_test)
                        self.set_cell_color(i, j, self.Color.WHITE)
                        
                        # Reinir the trace
                        self.column_validity_trace = [[None] * len(self.lSeq) for i in range(len(self.cSeq[j]) + 1)]
                        
                        success_white_coloring = self.compute_validity_trace(instance_height - 1, j, len(self.cSeq[j]), self.For.COLUMN)
                            
                        #success_white_coloring = (line_test and column_test)
                        had_change = False
                        
                        if not success_black_coloring and success_white_coloring:    
                            self.set_cell_color(i, j, self.Color.WHITE)
                            fixed_cells_count += 1
                            had_change = True
                        elif success_black_coloring and not success_white_coloring:
                            self.set_cell_color(i, j, self.Color.BLACK)
                            fixed_cells_count += 1
                            had_change = True
                        elif not success_black_coloring and not success_white_coloring:
                            return "ERROR"
                        else:
                            self.set_cell_color(i, j, self.Color.UNDEFINED)
                            
                        if had_change and not i in line_to_explore:
                            line_to_explore.append(i)
                            
            column_to_explore = []

# A solver using MIP (PLNE in french) to solve the grid.          
class LinearSolver:
    # A simple enumeration to determine the color of the cells
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
    
    # Load an instance from a text file and perform the necessary setup.    
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

                
        f.close()
    
    # Build the model variables from the instance parameters and sequences.
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
                    if (j < acc + t) or (j > self.instance_width - (seq_sum - acc +  num_seq - t - 1)):
                        y[l + j].append(None)
                        acc += st
                        continue
                    else:
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
            
    # Solve the nonogram using MIP    
    def solve(self, use_presolving = False, seed = 35594):
        partial_solution = None
        
        # If we're using presolving, we try to solve what we can with the dynamic programming
        # approach, and then fix the variable accordingly.
        if use_presolving:
            presolver = DynamicSolver()
            presolver.load_instance(self.instance_file)
            presolver.solve()
            partial_solution = presolver.instance
            
        x, y, z = self.build_model_variables(partial_solution)
        
        self.model.update()

        
        # Builds all the constraints for the lines.
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
                        self.model.addConstr(quicksum(y[l + k][t + 1] for k in range(j + st + 1, self.instance_width) if not y[l + k][t + 1] is None) >= seq_var)
            if not len(y[l]) == 0:
                for t in range(max_t + 1):
                    self.model.addConstr(quicksum(y[l + k][t] for k in range(0, self.instance_width) if not y[l + k][t] is None) == 1)
        
        # Builds all the contraints for the columns.                
        for j in range(self.instance_width):
            c = j * self.instance_height
            self.model.addConstr(quicksum(x[j + k * self.instance_width] for k in range(self.instance_height)) == sum(self.cSeq[j]))
            
            max_t = 0
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

        # Objective : maximizing the x_ij.
        self.model.setObjective(quicksum(x[idx] for idx in range(len(x))) ,GRB.MAXIMIZE)
        #self.model.update()
        self.model.setParam(GRB.Param.Seed, seed)
        self.model.write("debug.lp")

        # Optimizes the model
        self.model.optimize()
        
        # Exracts the variables' values.
        self.instance = [[x[idx].x for idx in range(l * self.instance_width, (l + 1) * self.instance_width)] for l in range(self.instance_height)]
        
            
                    

# An example using the linear programming solver to solve the instance 15.
# The dynamic solver works in almost exactly the same way, without any 
# arguments inside the "solve()" method.
solver = LinearSolver()
solver.load_instance("instances/14.txt")
#solver.solve()
solver.solve(True, 21488320)


solver.show_grid()
#21488320
#