# -*- coding: utf-8 -*-
import os
import sys
import csv
import subprocess
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
#sns.set_style("darkgrid")

from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D


def inputfile_listing():
	cwd = os.getcwd()	#current working directory
	dirs = []
	for x in os.listdir(cwd):
		if os.path.isdir(x):
			dirs.append(x)
		
	#ディレクトリ内をチェックし csv ファイルをfilesに追加
	files = []
	for dirname in dirs:
		path = cwd+"/"+dirname
		for y in os.listdir(path):
			base, ext = os.path.splitext(y)
			if ext=='.csv':
				files.append(cwd+"/"+dirname+"/"+y)
				
	number=np.arange(len(files))
	files_with_no = np.hstack([np.vstack(number), np.vstack(files)])
	
	return files_with_no

def inputfile_selection():
	# development mode ##
	#return "C:/Users/tmem/Desktop/DEMO_&_EXP/EXP_fixedpoint/query0001/log2_v13.1GD10_0.csv", "log2_v13.1GD10_0.csv"
	######
	
	for x in range(len(files)):
		print(files_with_no[x,0], ": ", files_with_no[x,1])
		
	print("Select the No.: ", end='')
	n=int(input())
	while n < 0 or len(files)-1 < n:
		print("Select number from the list above!: ", end='')
		n=int(input())

	return files_with_no[n,1]
		
def read_txt(fullname):
	data = np.loadtxt(fullname, delimiter=',',skiprows=3,
					usecols=(0,1,2,3,4,5,6,7,8,9,10,11)
					)
	return data

def get_GT_position(f_dir):
	
	if f_dir=="query0001":
		x, y, z, th, ph =0.3, 0.7, 1.2, 5.7, 0.0
	elif f_dir=="query0002":
		x, y, z, th, ph =0.3, 0.7, 1.2, 16.1, 0.0
	elif f_dir=="query0003":
		x, y, z, th, ph =0.3, 0.7, 1.2, -21.6, 0.0
	elif f_dir=="query0004":
		x, y, z, th, ph =0.3, 0.7, 1.2, -47.2, 0.0
	elif f_dir=="query0005":
		x, y, z, th, ph =0.3, 0.7, 1.2, 157.2, 0.0
	elif f_dir=="query0006":
		x, y, z, th, ph =0.3, 0.7, 1.2, 173.1, 0.0

	return x, y, z, th, ph
	
def get_DB_range(d):
	Xmin, Xmax = min(d[:,1]), max(d[:,1])
	Ymin, Ymax = min(d[:,2]), max(d[:,2])
	Zmin, Zmax = min(d[:,3]), max(d[:,3])
	Tmin, Tmax = min(d[:,4]), max(d[:,4])
	Pmin, Pmax = min(d[:,5]), max(d[:,5])
	return Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, Tmin, Tmax, Pmin, Pmax

def make_graph(d, f_dir, fname):
	gtx, gty, gtz, gtth, gtph = get_GT_position(f_dir)	#f_dir から真値座標を得る
	R = get_DB_range(d)
	
	fig = plt.figure(figsize=(18,6))
	ax1 = fig.add_subplot(121,projection='3d')
	
	ax1.set_xlim(R[0],R[1])
	ax1.set_ylim(R[2],R[3])
	ax1.set_zlim(0.0,1.0)
	ax1.set_xlabel("X")
	ax1.set_ylabel("Y")
	ax1.set_zlabel("sim index")
	#ax1.set_title(f_dir + " " +fname)
	
	ax1.plot([gtx,gtx],[gty,gty],[0,1], linestyle='solid', color ='red')
	ax1.plot([0,gtx],[gty,gty],[0,0], linestyle='dashed', color ='red')
	ax1.plot([gtx,gtx],[0,gty],[0,0], linestyle='dashed', color ='red')
	p1=ax1.scatter(d[:,1],d[:,2],d[:,11], 
					c=d[:,9],
					cmap='Blues',  # hot, Blues, YlOrBr, YlGn
					norm=Normalize(vmin=0, vmax=1)
					)
	fig.colorbar(p1)
	ax1.scatter(gtx,gty,1.0,c='red',s=50, marker='*')	#真値座標における sim index
	ax1.view_init(30, -120)
	
	ax2 = fig.add_subplot(122,projection='3d')
	ax2.set_xlim(R[0],R[1])
	ax2.set_ylim(R[2],R[3])
	ax2.set_zlim(-180,180)
	ax2.set_zticks([-180, -120, -60, 0, 60, 120, 180])
	ax2.set_xlabel("X")
	ax2.set_ylabel("Y")
	ax2.set_zlabel("best theta")
	#ax2.set_title(f_dir + " " +fname)

	ax2.plot([gtx,gtx],[gty,gty],[-180,gtth], linestyle='solid', color ='red',alpha=0.8)
	p2=ax2.scatter(d[:,1],d[:,2],d[:,4], 
					c=d[:,4],
					s=10,
					alpha=0.8,	  # マーカーの色を透過させる
					cmap='rainbow', # hot, Blues, YlOrBr, YlGn, gnuplot, hsv
					norm=Normalize(vmin=-180, vmax=180)
					)
	fig.colorbar(p2)
	ax2.scatter(gtx,gty,gtth,c='red',s=200, marker='*')	#真値座標の表示
	ax2.view_init(30, -120)
	
	#plt.savefig("best_th/(" + f_dir + ")" + fname + ".png")	# 上書き保存されない
	#plt.savefig("(" + f_dir + ")" + fname + ".png")			# 上書き保存される

	i=0
	for i, angle in enumerate(range(-180, -90)):
		ax1.view_init(30, angle)
		ax2.view_init(30, angle)
		plt.draw()
		plt.savefig("Animation/images/" + str(i) + ".jpg")
	
	command = "ffmpeg -r 10 -i Animation/images/%d.jpg -pix_fmt rgb24 -f gif Animation/out.gif"
	os.system(command)
	print("(" + f_dir + ")" + fname)
	
def best_th(d):		# それぞれの座標(x,y)で　sim_index が最大となる th　を抜き出して data を作成
	R, C =d.shape
	print(R,C)
	
	d_out = np.empty((0,12))	# 出力用配列の準備
		
	vX = np.unique(d[:,1])
	vY = np.unique(d[:,2])
	
	for x in vX:
		for y in vY:
			d1=d[np.where(d[:,1]==x)]	# d から 1列目=x の行を抜き出し d1 とする
			d2=d1[np.where(d1[:,2]==y)]	# d1 から 2列目=y の行を抜き出し d2 とする			
			d3=d2[np.argmax(d2[:,11]),:]	# d2 から　sim_index が最大の行を抜き出し d3 とする
			d_out=np.vstack((d_out,d3))

	return d_out
	
def main():
	files_w_n = inputfile_listing()
	end=0
	while(end >= 0):
		for x in range(len(files_w_n[:,0])):
			print(files_w_n[x,0], ": ", files_w_n[x,1])
		
		print("Insert -1 to stop the process")
		print("Select the No.: ", end='')
		n=int(input())
		if (n<0): break
		
		fullname = files_w_n[n,1]
	
		p=fullname.find('EXP_fixedpoint')
		fname = fullname[p+25:]		#fname
		f_dir = fullname[p+15:p+24]	#p_directory

		data = read_txt(fullname)
		d_best_th = best_th(data)
		make_graph(d_best_th, f_dir, fname)

if __name__ == "__main__":
	main()