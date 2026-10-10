import numpy as np
import matplotlib.pyplot as plt

def create_maze(SIZE):
    internal_rows=SIZE if SIZE%2==1 else SIZE-1
    internal_cols=SIZE if SIZE%2==1 else SIZE-1
    
    grid=np.ones((internal_rows+2, internal_cols+2))
    
    # 初始化起点
    start=(internal_rows, 1)
    grid[start]=0
    stack=[start]
    
    directions=[(-2, 0), (2, 0), (0, -2), (0, 2)]
    
    while stack:
        x, y=stack[-1]
        neighbors=[]
        for dx, dy in directions:
            nx, ny=x+dx, y+dy
            if 1<=nx<=internal_rows and 1<=ny<=internal_cols and grid[nx, ny]==1:
                neighbors.append((nx, ny))
        
        if neighbors:
            nx, ny=neighbors[np.random.randint(len(neighbors))]
            grid[nx, ny]=0
            grid[(nx+x)//2, (ny+y)//2]=0
            stack.append((nx, ny))
        else:
            stack.pop()
    
    if SIZE%2==0:
        new_grid=np.ones((SIZE+2, SIZE+2))
        new_grid[1:internal_rows+1, 1:internal_cols+1]=grid[1:internal_rows+1, 1:internal_cols+1]
        
        # 在右侧添加一列通道
        for i in range(1, SIZE+1):
            if i%2==1:
                new_grid[i, internal_cols+1]=0
                if internal_cols+1<=SIZE and new_grid[i, internal_cols]==0:
                    new_grid[i, internal_cols]=0
        
        # 在下侧添加一行通道
        for j in range(1, SIZE+1):
            if j%2==1:
                new_grid[internal_rows+1, j]=0
                if internal_rows+1<=SIZE and new_grid[internal_rows, j]==0:
                    new_grid[internal_rows, j]=0
        
        grid=new_grid
    
    return grid

SIZE=16
grid=create_maze(SIZE)

# 绘制迷宫俯视图
plt.figure(figsize=(SIZE, SIZE))
plt.imshow(grid, cmap='binary', interpolation='nearest')
plt.xticks([])
plt.yticks([])
plt.grid(True)

# 标记入口和出口
plt.text(1, SIZE, 'Enter', ha='center', va='center', color='r', fontsize=12, fontweight='bold')
plt.text(SIZE, 1, 'Leave', ha='center', va='center', color='g', fontsize=12, fontweight='bold')
plt.title(f'Maze Size: {SIZE}x{SIZE}')
    
plt.tight_layout()
plt.savefig(f'Maze{SIZE}x{SIZE}.png', dpi=300, bbox_inches='tight')
plt.close()