from turtle import distance
import imageio
import sys
case = sys.argv[1]
frame_num = int(sys.argv[2])

images = []
for j in range(frame_num):
    print(j)
    images.append(imageio.imread('job/'+case+'/output/'+str(j).zfill(3)+'_flowfield_yz.png'))
imageio.mimsave(case+'_yz.gif',images,fps=20)

images = []
for j in range(frame_num):
    print(j)
    images.append(imageio.imread('job/'+case+'/output/'+str(j).zfill(3)+'_flowfield_xz.png'))
imageio.mimsave(case+'_xz.gif',images,fps=20)
