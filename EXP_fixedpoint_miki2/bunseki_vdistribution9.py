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
import time
#sns.set_style("darkgrid")

from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

workspace = 'EXP_fixedpoint_miki2'

def inputfile_listing():
	cwd = os.getcwd()	#current working directory
	dirs = []
	for x in os.listdir(cwd):
		if os.path.isdir(x):
			dirs.append(x)

	path = cwd+"\\u_result" #UMap推定結果の.csvファイルを読み込む
	#print(path)

	#ディレクトリ内をチェックし csv ファイルをfilesに追加
	files = []
	results = []
	distribution = []

	path = cwd+"\\"+ "u_result"
	for f in os.listdir(path):
		#print("f: {}".format(f))
		base, ext = os.path.splitext(f)
		if ext=='.csv':
			files.append(path + "\\" + f)
	number_files =np.arange(len(files))
	files_with_no = np.hstack([np.vstack(number_files), np.vstack(files)])

	#files[]にはu_result内のUMapマッチング結果のcsvが取り込まれる
	#files_with_no = np.hstack([np.vstack(number), np.vstack(files)])

	#for x in range(len(files)):
	#	print(files[x])

	#simresultデイレクトリ内のcsvファイルを取り込む
	result_file_dir = cwd + "/simresult"
	for y in os.listdir(result_file_dir):
		base, ext = os.path.splitext(y)
		if ext=='.csv':
			results.append(result_file_dir+"/"+y)

	number_result=np.arange(len(results))
	results_with_no = np.hstack([np.vstack(number_result), np.vstack(results)])

	#normal_distributionデイレクトリ内のcsvファイルを取り込む
	distribution_file_dir = cwd + "/normal_distribution"
	for z in os.listdir(distribution_file_dir):
			base, ext = os.path.splitext(z)
			if ext=='.csv':
				distribution.append(distribution_file_dir+"/"+z)

	number=np.arange(len(distribution))
	distribution_with_no = np.hstack([np.vstack(number), np.vstack(distribution)])

	return files_with_no, results_with_no, distribution_with_no


def read_txt(fullname):
	#3行をスキップして0~13列までのデータをdataへ格納する
	data = np.loadtxt(fullname, delimiter=',',skiprows=3,
					usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
					)
	return data

def read_resulttxt(fullname):
    #1行をスキップして0~16列までのデータをdataへ格納する
    data = np.loadtxt(fullname, delimiter=',',skiprows=1,
				usecols=(0,1,2,3,4,5,6,7,8,9)
				)

    return data

def read_disttxt(distname):
    #1行をスキップして0~16列までのデータをdataへ格納する
    data = np.loadtxt(distname, delimiter=',',skiprows=1,
				usecols=(1)
				)

    return data
def read_dist2txt(distname):
    #1行をスキップして0~16列までのデータをdataへ格納する
    data = np.loadtxt(distname, delimiter=',',skiprows=1,
				usecols=(2)
				)

    return data

def read_dist3txt(distname):
    #1行をスキップして0~16列までのデータをdataへ格納する
    data = np.loadtxt(distname, delimiter=',',skiprows=1,
				usecols=(3)
				)

    return data

def read_dist5txt(distname):
    #1行をスキップして0~16列までのデータをdataへ格納する
    data = np.loadtxt(distname, delimiter=',',skiprows=1,
				usecols=(0,1,2,3,4,)
				)

    return data

def get_query_info(fullname):
	f=open(fullname)
	line = f.readline()
	line = f.readline()	# 2行目を読み込んで、line を上書き
	f.close
	line = "10" + line
	query = np.array(line.split(','), dtype = np.float)

	return query

#query画像の真値座標を定義
def get_GT_position(f_dir, query_id, res_data, n):
	#query画像ごとに真値座標は変化する必要がある
	#x, y, z, th, ph = res_data[query_id, 1], res_data[query_id, 2], 0.2, res_data[query_id, 3] , 0
	x, y, z, th, ph = res_data[n, 1], res_data[n, 2], 0.2, res_data[n, 3] , 0
	query = np.array([x,y,z,th,ph], dtype = np.float)

	return x, y, z, th, ph, query

def get_DB_range(d):
	Xmin, Xmax = min(d[:,1]), max(d[:,1])
	Ymin, Ymax = min(d[:,2]), max(d[:,2])
	Zmin, Zmax = min(d[:,3]), max(d[:,3])
	Tmin, Tmax = min(d[:,4]), max(d[:,4])
	Pmin, Pmax = min(d[:,5]), max(d[:,5])
	return Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, Tmin, Tmax, Pmin, Pmax

def calc_distance(q, db):

	distance = np.zeros(len(db[:,0]))
	norm = np.zeros(len(db[:,0]))
	def_ang = np.zeros(len(db[:,0])) 	#deflection angle 偏角
	print("q[0] : {}".format(q[0]))
	print("q[1] : {}".format(q[1]))
	print("q[2] : {}".format(q[2]))
	print("q[3] : {}".format(q[3]))

	for i, dummy in enumerate(db[:,0]):
		#distance[i] = np.linalg.norm( db[i,1:4:1] - q[1:4:1] )

		a = np.array([db[i,1],db[i,2],db[i,3]])
		b = np.array([q[0],q[1],q[2]])
		distance[i] = np.linalg.norm( a - b )

		c = np.array((db[i,1],db[i,2],db[i,3], db[i,4]*np.pi/180, db[i,5]*np.pi/180))
		d = np.array([q[0],q[1],q[2],q[3]*np.pi/180,q[4]*np.pi/180])
		norm[i] = np.linalg.norm( c - d )
		#print("norm[i] : {}".format(norm[i]))


		e = np.array((db[i,4]*np.pi/180, db[i,5]*np.pi/180))
		f = np.array([q[3]*np.pi/180,q[4]*np.pi/180])
		def_ang[i] = np.linalg.norm( e - f )

	#print(q[1:4:1])
	print (i)
	print("distance[] : {}".format(len(distance)))
#	normwithno = np.hstack([np.vstack(db), np.vstack(norm)])
	return distance, norm, def_ang

#def norm_array(norm, q):
#	dict

def calc_distance2(q, db):
	norm = np.zeros(len(db[:,0]))
	l=0

	for i, dummy in enumerate(db[:,0]):
		for j, dummy in enumerate(db[:,0]):
			for k, dummy in enumerate(db[:,0]):
				c = np.array((db[i,1],db[j,2],db[k,4]*np.pi/180))
				d = np.array(q[0],q[1],q[3]*np.pi/180)
				norm[l] = np.linalg.norm( c - d )
				l = l+1

	return norm

def min_value(norm):
	min = norm.argmin()
	min_i = 0
	for i in range(len(norm)):
		if norm[i] == min :
			min = norm[i]
			min_i = i

	return min, min_i

def normgraph(norm, value):
	#value グラフの横軸を誤差が小さい順に並べ替える
	val_zip = dict(zip(norm,value))
	s_val = sorted(val_zip.items(), key=lambda x:x[0])
	myx = []
	myy = []
	for x in range(len(s_val)):
		myx.append(s_val[x][0])
	for y in range(len(s_val)):
		myy.append(s_val[y][1])

	myymax = 0
	maxi = 0
	for i in range(len(myy)):
		if myy[i] > myymax :
			myymax = myy[i]
			maxi = i

	maxmyy = max(myy)
	return myx, myy, maxmyy, maxi

def plot_values(d, q, dist, n, def_ang, f_dir, fname):
	bestid = n.argmin()

	A = d[:,6]
	B = d[:,7]
	Bc= d[:,8]/255
	C = d[:,9]
	D = d[:,10]
	E = d[:,11]
	Ec= d[:,12]/255
	F = d[:,13]

	fig = plt.figure(figsize=(20,12))	#figsize=（横幅、縦幅）

	ax1 = fig.add_subplot(5,4,1)
	ax1.set_title("query=("+str(q[1])+","+str(q[2])+","+str(q[3])+"),("+str(q[4])+","+str(q[5])+")" ,size=12,color="black")
	ax1.plot(d[:,6]+d[:,7]+d[:,9], linestyle='solid',lw=0.1,  color ='black', label="n(A)+n(B)+n(C)")
	plt.legend(loc="lower right")

	ax2 = fig.add_subplot(5,4,2)
	ax2.plot(d[:,9], linestyle='solid', lw=0.1, color ='limegreen', label="n(C)")
	p=d[:,9].argmax()
	ax2.text(0,1000, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")" ,size=10,color="black")
	ax2.text(0,500, "dist="+str(round(dist[p],4)),size=10,color="black")
	ax2.text(0,0.00, "norm="+str(round(n[p],3)),size=10,color="black")
	plt.legend(loc="lower right")
	ax2.plot([bestid,bestid],[0,max(d[:,9])], linestyle='solid',lw=0.3, color ='red')
	ax2.plot([p,p],[0,max(d[:,9])], linestyle='solid',lw=0.3, color ='blue')

	ax5 = fig.add_subplot(545)
	ax5.plot(d[:,9]+d[:,10], linestyle='solid',lw=0.1,  color ='indianred', label="n(C)+n(D)")
	plt.legend(loc="lower right")
	ax5.plot([bestid,bestid],[0,max(d[:,9]+d[:,10])], linestyle='solid',lw=0.3, color ='red')

	ax6 = fig.add_subplot(5,4,6)
	ax6.plot(d[:,7]+d[:,9], linestyle='solid',lw=0.1,  color ='limegreen', label="n(B)+n(C)")	# graded dilation matching
	p=np.argmax(d[:,8]/255+d[:,9])
	ax6.text(0,2000, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")" ,size=10,color="black")
	ax6.text(0,1000, "dist="+str(round(dist[p],4)),size=10,color="black")
	ax6.text(0,0.0, "norm="+str(round(n[p],3)),size=10,color="black")
	plt.legend(loc="lower right")
	ax6.plot([bestid,bestid],[0,max(d[:,7]+d[:,9])], linestyle='solid',lw=0.3, color ='red')


	ax9 = fig.add_subplot(549)
	ax9.plot(d[:,9]+d[:,10]+(d[:,8]+d[:,12])/255, linestyle='solid',lw=0.1,  color ='indianred', label="c(B)+n(C)+n(D)+c(E)")
	plt.legend(loc="lower right")
	ax9.plot([bestid,bestid],[0,30000], linestyle='solid',lw=0.3, color ='red')

	ax10 = fig.add_subplot(5,4,10)
	ax10.plot(d[:,8]/255+d[:,9], linestyle='solid',lw=0.1,  color ='limegreen', label="c(B)+n(C)")	# graded dilation matching
	p=np.argmax(d[:,8]/255+d[:,9])
	ax10.text(0,2000, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")" ,size=10,color="black")
	ax10.text(0,1000, "dist="+str(round(dist[p],4)),size=10,color="black")
	ax10.text(0,0.0, "norm="+str(round(n[p],3)),size=10,color="black")
	plt.legend(loc="lower right")
	ax10.plot([bestid,bestid],[0,max(d[:,8]/255+d[:,9])], linestyle='solid',lw=0.3, color ='red')

	ax11 = fig.add_subplot(5,4,11)
	si0=d[:,9] / ( d[:,9]+d[:,10] )
	p=si0.argmax()
	ax11.plot(si0, linestyle='solid',lw=0.1,  color ='violet', label="si0")	 #sim index
	ax11.set_ylim(0,1)
	ax11.plot([bestid,bestid],[0,1], linestyle='solid',lw=0.3, color ='red')
	ax11.plot(p,si0.max(), 'o', markersize=10, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	ax11.text(0,0.16, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")" ,size=10,color="black")
	ax11.text(0,0.08, "dist="+str(round(dist[p],4)),size=10,color="black")
	ax11.text(0,0.00, "norm="+str(round(n[p],3)),size=10,color="black")
	plt.legend(loc="lower right")

	#################################
	ax12 = fig.add_subplot(5,4,12)
	si4=d[:,9] / ( d[:,9]+d[:,10] )
	p=si4.argmax()
	ax12.plot(si4, linestyle='solid',lw=0.1,  color ='violet', label="si4")	 #sim index
	ax12.set_ylim(0,1)
	ax12.plot([bestid,bestid],[0,1], linestyle='solid',lw=0.3, color ='red')
	ax12.plot(p,si4.max(), 'o', markersize=10, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	ax12.text(0,0.16, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")" ,size=10,color="black")
	ax12.text(0,0.08, "dist="+str(round(dist[p],4)),size=10,color="black")
	ax12.text(0,0.00, "norm="+str(round(n[p],3)),size=10,color="black")
	plt.legend(loc="lower right")
	##################################

	ax13 = fig.add_subplot(5,4,13)
	ax13.plot(d[:,7]+d[:,11], linestyle='solid',lw=0.1,  color ='olive', label="n(B)+n(E)")	# graded dilation matching
	p=np.argmax(d[:,8]/255+d[:,9])
	plt.legend(loc="lower right")
	ax13.plot([bestid,bestid],[0,max(d[:,7]+d[:,11])], linestyle='solid',lw=0.3, color ='red')

	ax14 = fig.add_subplot(5,4,14)
	ax14.plot(d[:,10], linestyle='solid',lw=0.1,  color ='olive', label="n(D)")	# graded dilation matching
	p=np.argmax(d[:,8]/255+d[:,9])
	plt.legend(loc="lower right")
	ax14.plot([bestid,bestid],[0,max(d[:,10])], linestyle='solid',lw=0.3, color ='red')

	ax15 = fig.add_subplot(5,4,15)
	si2=(d[:,8]/255+d[:,9]) / (d[:,9]+d[:,10]+(d[:,8]+d[:,12])/255)
	p=si2.argmax()
	ax15.plot( si2, linestyle='solid',lw=0.1,  color ='violet', label="si2")
	ax15.set_ylim(0,1)
	ax15.plot([bestid,bestid],[0,1], linestyle='solid',lw=0.3, color ='red')
	ax15.text(0,0.18, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")" ,size=10,color="black")
	ax15.text(0,0.10, "dist="+str(round(dist[p],4)),size=10,color="black")
	ax15.text(0,0.02, "norm="+str(round(n[p],3)),size=10,color="black")
	plt.legend(loc="lower right")
	ax15.plot(p,si2.max(), 'o', markersize=10, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)

	ax16 = fig.add_subplot(5,4,16)
	si3=(d[:,8]/255+d[:,9]) / (d[:,7]+d[:,9])
	p=si3.argmax()
	ax16.plot( si3, linestyle='solid',lw=0.1,  color ='violet', label="si3")	 #sim index
	ax16.set_ylim(0,1)
	ax16.plot([bestid,bestid],[0,1], linestyle='solid',lw=0.3, color ='red')
	ax16.text(0,0.18, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")" ,size=10,color="black")
	ax16.text(0,0.10, "dist="+str(round(dist[p],4)),size=10,color="black")
	ax16.text(0,0.02, "norm="+str(round(n[p],3)),size=10,color="black")
	ax16.plot(p,si3.max(), 'o', markersize=10, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	plt.legend(loc="lower right")

	ax17 = fig.add_subplot(5,4,17)
	ax17.plot(d[:,11], linestyle='solid',lw=0.1,  color ='olive', label="n(E)")
	p=np.argmax(d[:,8]/255+d[:,9])
	plt.legend(loc="lower right")
	ax17.plot([bestid,bestid],[0,15000], linestyle='solid',lw=0.3, color ='red')

	ax18 = fig.add_subplot(5,4,18)
	ax18.plot(d[:,10]+d[:,11], linestyle='solid',lw=0.1,  color ='olive', label="n(D)+n(E)")
	p=np.argmax(d[:,8]/255+d[:,9])
	plt.legend(loc="lower right")
	ax18.plot([bestid,bestid],[0,15000], linestyle='solid',lw=0.3, color ='red')

	ax19 = fig.add_subplot(5,4,19)
	si1=(d[:,8]/255+d[:,9]) / ( d[:,6]+d[:,7]+d[:,9] ) # (Bcon+C*255) / ( (A+B+C) * 255)
	p=si1.argmax()
	ax19.plot(si1, linestyle='solid',lw=0.1,  color ='violet', label="si1")	 #sim index
	ax19.set_ylim(0,1)
	ax19.plot([bestid,bestid],[0,1], linestyle='solid',lw=0.3, color ='red')
	ax19.plot(p,si1.max(), 'o', markersize=10, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	plt.legend(loc="lower right")

	ax20 = fig.add_subplot(5,4,20)
	#si5=(d[:,8]/255+d[:,9]) / ( d[:,6]+d[:,7]+d[:,9]) -  d[:,10]/(d[:,9]+d[:,10])
	si5=(d[:,8]/255+d[:,9]) / ( d[:,6]+d[:,7]+d[:,9]+d[:,10])
	ax20.plot(si5, linestyle='solid',lw=0.1,  color ='violet', label="si5")
	p=si5.argmax()
	print(p,si5.max())
	ax20.text(0,1.6, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")" ,size=10,color="black")
	ax20.text(0,0.8, "dist="+str(round(dist[p],4)),size=10,color="black")
	ax20.text(0,0.0, "norm="+str(round(n[p],3)),size=10,color="black")
	plt.legend(loc="lower right")
	ax20.set_ylim(0,1)
	ax20.plot([bestid,bestid],[0,1], linestyle='solid',lw=0.3, color ='red')
	ax20.plot(p,si5.max(), 'o', markersize=10, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)

	#ax20.plot([p,p],[0,1], linestyle='solid',lw=0.3, color ='blue')

	###############
	###############
	ax443 = fig.add_subplot(5,4,3)
	p=n.argmin()
	ax443.plot(dist, linestyle='solid',lw=0.1,  color ='black', label="dist")	 #norm
	plt.legend(loc="upper right")
	ax443.set_ylim(0,np.pi)
	ax443.plot([p,p],[0,np.pi], linestyle='solid',lw=0.3, color ='red')

	ax444 = fig.add_subplot(5,4,4)
	p=n.argmin()
	ax444.plot(n, linestyle='solid',lw=0.1,  color ='black', label="norm")	 #norm
	plt.legend(loc="upper right")
	ax444.set_ylim(0,np.pi)
	ax444.plot([p,p],[0,np.pi], linestyle='solid',lw=0.3, color ='red')

	ax447 = fig.add_subplot(5,4,7)
	p=n.argmin()
	ax447.plot(dist, linestyle='solid',lw=0.1,  color ='black', label="dist(zoom)")
	plt.legend(loc="upper right")
	ax447.set_xlim(bestid-149,bestid+150)
	ax447.set_ylim(0,0.2)
	ax447.plot([p,p],[0,np.pi], linestyle='solid',lw=0.3, color ='red')

	ax448 = fig.add_subplot(5,4,8)
	p=n.argmin()
	ax448.plot(n, linestyle='solid',lw=0.1,  color ='black', label="norm(zoom)")	 #norm
	plt.legend(loc="upper right")
	ax448.set_xlim(bestid-149,bestid+150)
	ax448.set_ylim(0,0.2)
	ax448.plot([bestid,bestid],[0,np.pi], linestyle='solid',lw=0.3, color ='red')
	ax448.text(bestid-149,0.07, "best-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")" ,size=12,color="black")
	ax448.text(bestid-149,0.04, "dist="+str(round(dist[p],4)),size=10,color="black")
	ax448.text(bestid-149,0.01, "norm="+str(round(n[p],3)),size=10,color="black")


	plt.show()

def plot_values2(d, q, dist, n, def_ang, f_dir, fname):
	Pbst = n.argmin()
	A = d[:,6]
	B = d[:,7]
	Bc= d[:,8]
	C = d[:,9]
	D = d[:,10]
	E = d[:,11]
	Ec= d[:,12]
	F = d[:,13]-3840 # CUDAマッチング時の都合上、320*12=3840 ピクセル数ほど 両ゼロピクセルが追加されているため

	fig = plt.figure(figsize=(20,12))	#figsize=（横幅、縦幅）

	ax1 = fig.add_subplot(5,4,1)
	ax1.set_title("query=("+str(q[1])+","+str(q[2])+","+str(q[3])+"),("+str(q[4])+","+str(q[5])+")" ,size=12,color="black")
	ax1.plot(A, linestyle='solid',lw=0.1,  color ='grey', label="n(A)")
	plt.legend(loc="lower right")
	ax1.plot([Pbst,Pbst],[0,A.max()], linestyle='solid',lw=0.3, color ='red')

	ax2 = fig.add_subplot(5,4,2)
	ax2.plot(B, linestyle='solid',lw=0.1,  color ='grey', label="n(B)")
	plt.legend(loc="lower right")
	ax2.plot([Pbst,Pbst],[0,B.max()], linestyle='solid',lw=0.3, color ='red')

	ax3 = fig.add_subplot(5,4,3)
	ax3.plot(Bc, linestyle='solid',lw=0.1,  color ='grey', label="c(B)")
	plt.legend(loc="lower right")
	ax3.plot([Pbst,Pbst],[0,Bc.max()], linestyle='solid',lw=0.3, color ='red')

	ax4 = fig.add_subplot(5,4,4)
	ax4.plot(C, linestyle='solid',lw=0.1,  color ='grey', label="n(C)")
	plt.legend(loc="lower right")
	p=C.argmax()
	ax4.set_title("pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+"), d:"+str(round(dist[p],4))+", n:"+str(round(n[p],3)) ,size=10,color="blue")
	ax4.plot([Pbst,Pbst],[0,C.max()], linestyle='solid',lw=0.3, color ='red')
	ax4.plot(p,C.max(), 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	ax4.text(0,0.01, "d:"+str(round(dist[p],4))+", n:"+str(round(n[p],3)),size=10,color="blue")


	ax5 = fig.add_subplot(5,4,5)
	ax5.plot(D, linestyle='solid',lw=0.1,  color ='grey', label="n(D)")
	plt.legend(loc="lower right")
	ax5.plot([Pbst,Pbst],[0,D.max()], linestyle='solid',lw=0.3, color ='red')

	ax6 = fig.add_subplot(5,4,6)
	ax6.plot(E, linestyle='solid',lw=0.1,  color ='grey', label="n(E)")
	plt.legend(loc="lower right")
	ax6.plot([Pbst,Pbst],[0,E.max()], linestyle='solid',lw=0.3, color ='red')

	ax7 = fig.add_subplot(5,4,7)
	ax7.plot(Ec, linestyle='solid',lw=0.1,  color ='grey', label="c(E)")
	plt.legend(loc="lower right")
	ax7.plot([Pbst,Pbst],[0,Ec.max()], linestyle='solid',lw=0.3, color ='red')

	ax8 = fig.add_subplot(5,4,8)
	ax8.plot(Bc+C, linestyle='solid',lw=0.1,  color ='grey', label="c(B)+n(C)")
	plt.legend(loc="lower right")
	ax8.plot([Pbst,Pbst],[0,max(Bc+C)], linestyle='solid',lw=0.3, color ='red')

	ax9 = fig.add_subplot(5,4,9)
	si0=C/(C+D+0.0001)
	p=si0.argmax()
	ax9.plot(si0, linestyle='solid',lw=0.1,  color ='grey', label="si0")
	plt.legend(loc="lower right")
	ax9.set_ylim(0,1.0)
	ax9.plot([Pbst,Pbst],[0,1], linestyle='solid',lw=0.3, color ='red')
	ax9.plot(p,si0[p], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	ax9.text(0,0.8, str(p),size=12,color="blue")
	ax9.text(0,0.3, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")",size=12,color="blue")
	ax9.text(0,0.1, "d:"+str(round(dist[p],4))+", n:"+str(round(n[p],3)),size=12,color="blue")

	ax10 = fig.add_subplot(5,4,10)
	si1=(Bc+C)/(A+B+C+0.0001)
	p=si1.argmax()
	ax10.plot(si1, linestyle='solid',lw=0.1,  color ='grey', label="si1")
	plt.legend(loc="lower right")
	ax10.set_ylim(0,1.0)
	ax10.plot([Pbst,Pbst],[0,1], linestyle='solid',lw=0.3, color ='red')
	ax10.plot(p,si1[p], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	ax10.text(0,0.8, str(p),size=12,color="blue")
	ax10.text(0,0.3, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")",size=12,color="blue")
	ax10.text(0,0.1, "d:"+str(round(dist[p],4))+", n:"+str(round(n[p],3)),size=12,color="blue")

	ax11 = fig.add_subplot(5,4,11)
	si2=(Bc+C)/(Bc+C+D+Ec+0.0001)
	p=si2.argmax()
	ax11.plot(si2, linestyle='solid',lw=0.1,  color ='grey', label="si2")
	plt.legend(loc="lower right")
	ax11.set_ylim(0,0.2)
	ax11.plot([Pbst,Pbst],[0,0.2], linestyle='solid',lw=0.3, color ='red')
	ax11.plot(p,si2[p], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	ax11.text(0,0.15, str(p),size=10,color="blue")
	ax11.text(0,0.1, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")",size=12,color="blue")
	ax11.text(0,0.0, "d:"+str(round(dist[p],4))+", n:"+str(round(n[p],3)),size=12,color="blue")

	ax12 = fig.add_subplot(5,4,12)
	si3=(Bc+C)/(A+Bc+C+D+Ec+0.0001)
	p=si3.argmax()
	ax12.plot(si3, linestyle='solid',lw=0.1,  color ='grey', label="si3")
	plt.legend(loc="lower right")
	ax12.set_ylim(0,0.2)
	ax12.plot([Pbst,Pbst],[0,0.4], linestyle='solid',lw=0.3, color ='red')
	ax12.plot(p,si3[p], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	ax12.text(0,0.15, str(p),size=10,color="blue")
	ax12.text(0,0.03, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")",size=12,color="blue")
	ax12.text(0,0.01, "d:"+str(round(dist[p],4))+", n:"+str(round(n[p],3)),size=12,color="blue")

	ax13 = fig.add_subplot(5,4,13)
	si4=(Bc+C-A)/(A+Bc+C+D+Ec+0.0001)
	p=si4.argmax()
	ax13.plot(si4, linestyle='solid',lw=0.1,  color ='grey', label="si4")
	plt.legend(loc="lower right")
	ax13.set_ylim(-0.3,0.3)
	ax13.plot([Pbst,Pbst],[0,0.4], linestyle='solid',lw=0.3, color ='red')
	ax13.plot(p,si4[p], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	ax13.text(0,0.15, str(p),size=10,color="blue")
	ax13.text(0,0.05, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")",size=12,color="blue")
	ax13.text(0,0.01, "d:"+str(round(dist[p],4))+", n:"+str(round(n[p],3)),size=12,color="blue")

	ax14 = fig.add_subplot(5,4,14)
	si5=(Bc+C-A-D)/(A+Bc+C+D+Ec+0.0001)
	p=si5.argmax()
	ax14.plot(si5, linestyle='solid',lw=0.1,  color ='grey', label="si5")
	plt.legend(loc="lower right")
	ax14.set_ylim(-0.3,0.3)
	ax14.plot([Pbst,Pbst],[0,0.4], linestyle='solid',lw=0.3, color ='red')
	ax14.plot(p,si5[p], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	ax14.text(0,0.15, str(p),size=10,color="blue")
	ax14.text(0,0.05, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")",size=12,color="blue")
	ax14.text(0,0.01, "d:"+str(round(dist[p],4))+", n:"+str(round(n[p],3)),size=12,color="blue")

	ax15 = fig.add_subplot(5,4,15)
	si2=(Bc+C)/(Bc+C+D+Ec+0.0001)
	p=si2.argmax()
	ax15.plot(si2, linestyle='solid',lw=0.1,  color ='grey', label="si2(zoom)")
	plt.legend(loc="lower right")
	ax15.set_ylim(0,0.15)
	ax15.set_xlim(Pbst-1049,Pbst+1050)
	ax15.plot([Pbst,Pbst],[0,0.15], linestyle='solid',lw=0.3, color ='red')
	ax15.plot(p,si2[p], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	ax15.text(Pbst-1049,0.03, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")",size=12,color="blue")
	ax15.text(Pbst-1049,0.01, "d:"+str(round(dist[p],4))+", n:"+str(round(n[p],3)),size=12,color="blue")

	ax16 = fig.add_subplot(5,4,16)
	si3=(Bc+C)/(A+Bc+C+D+Ec+0.0001)
	p=si3.argmax()
	ax16.plot(si3, linestyle='solid',lw=0.1,  color ='grey', label="si3(zoom)")
	plt.legend(loc="lower right")
	ax16.set_xlim(Pbst-1049,Pbst+1050)
	ax16.set_ylim(0,0.15)
	ax16.plot([Pbst,Pbst],[0,0.15], linestyle='solid',lw=0.3, color ='red')
	ax16.plot(p,si3[p], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	ax16.text(Pbst-1049,0.03, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")",size=12,color="blue")
	ax16.text(Pbst-1049,0.01, "d:"+str(round(dist[p],4))+", n:"+str(round(n[p],3)),size=12,color="blue")


	#ax18 = fig.add_subplot(5,4,18)
	#ax18.plot(A+B+C+D+E+F, linestyle='solid',lw=0.1,  color ='grey', label="n(A)+n(B)+n(C)+n(D)+n(E)+n(F)")
	#plt.legend(loc="lower right")
	#ax18.plot([Pbst,Pbst],[0,Ec.max()], linestyle='solid',lw=0.3, color ='red')

	#print(A[200]+B[200]+C[200]+D[200]+E[200]+F[200])


	ax19 = fig.add_subplot(5,4,19)
	ax19.plot(n, linestyle='solid',lw=0.1,  color ='purple', label="norm")	 #norm
	plt.legend(loc="upper right")
	ax19.set_ylim(0,n.max())
	ax19.text(0,0.1, str(Pbst),size=10,color="red")
	ax19.plot([Pbst,Pbst],[0,n.max()], linestyle='solid',lw=0.3, color ='red')


	ax20 = fig.add_subplot(5,4,20)
	ax20.plot(n, linestyle='solid',lw=0.1,  color ='purple', label="norm(zoom)")	 #norm
	plt.legend(loc="upper right")
	ax20.set_xlim(Pbst-149,Pbst+150)
	ax20.set_ylim(0,0.2)
	#ax20.set_title("best-posi=("+str(d[Pbst,1])+","+str(d[Pbst,2])+","+str(d[Pbst,3])+"),("+str(d[Pbst,4])+","+str(d[Pbst,5])+"), d:"+str(round(dist[Pbst],4))+", n:"+str(round(n[Pbst],3)) ,size=10,color="red")
	ax20.plot([Pbst,Pbst],[0,n.max()], linestyle='solid',lw=0.3, color ='red')
	ax20.text(0,0.15, str(Pbst),size=12,color="red")
	ax20.text(Pbst-140,0.1, "best-posi=("+str(d[Pbst,1])+","+str(d[Pbst,2])+","+str(d[Pbst,3])+"),("+str(d[Pbst,4])+","+str(d[Pbst,5])+")" ,size=12,color="red")
	ax20.text(Pbst-140,0.07, "d:"+str(round(dist[Pbst],4))+", n:"+str(round(n[Pbst],3)),size=12,color="red")
	#ax20.text(Pbst-149,0.01, "norm="+str(round(n[Pbst],3)),size=10,color="red")

	#plt.savefig("(" + f_dir + ")" + fname+".jpg",format='jpg',dpi=200)
	plt.show()

def plot_values3(d, q, dist, n, def_ang, f_dir, fname):
	Pbst = n.argmin()#normの最小値
	A = d[:,6]
	B = d[:,7]
	Bc= d[:,8]
	C = d[:,9]
	D = d[:,10]
	E = d[:,11]
	Ec= d[:,12]
	F = d[:,13]-3840 # CUDAマッチング時の都合上、320*12=3840 ピクセル数ほど 両ゼロピクセルが追加されているため

	fig = plt.figure(figsize=(20,12))	#figsize=（横幅、縦幅）

  #queryimgのピクセル数
	ax1 = fig.add_subplot(1,2,1)
	ax1.set_title("query=("+str(q[0])+","+str(q[1])+","+str(q[2])+"),("+str(q[3])+","+str(q[4])+")" ,size=12,color="black")
	ax1.plot(A, linestyle='solid',lw=0.1,  color ='grey', label="n(A)")
	plt.legend(loc="lower right")
	ax1.plot([Pbst,Pbst],[0,A.max()], linestyle='solid',lw=0.3, color ='red')

  #類似度グラフ
	ax11 = fig.add_subplot(1,2,2)#グラフを描画する場所を定義
	si=(Bc+C)/(A+B+C+D+E+0.0001) #類似度の評価方法を定義
	p=si.argmax()
	ax11.plot(si, linestyle='solid',lw=0.1,  color ='grey', label="si")
	plt.legend(loc="lower right")
	ax11.set_ylim(0,0.2)
	ax11.plot([Pbst,Pbst],[0,0.2], linestyle='solid',lw=0.3, color ='red')#真値座標を定義
	ax11.plot(p,si[p], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	ax11.text(0,0.15, str(p),size=10,color="blue") #推定位置座標
	ax11.text(0,0.1, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")",size=12,color="blue")
	ax11.text(0,0.0, "d:"+str(round(dist[p],4))+", n:"+str(round(n[p],3)),size=12,color="blue")

	r_file = "C:\\Users\\k0913\\Desktop\\EXP_&_DEMO\\EXP_fixedpoint_try\\result_image"

	fig.savefig(r_file + "\\" + "img.png")

	plt.show()#グラフを表示
def plot_values4(d, q, dist, n, def_ang, f_dir, fname):
	Pbst = n.argmin()#normの最小値
	A = d[:,6]
	B = d[:,7]
	Bc= d[:,8]
	C = d[:,9]
	D = d[:,10]
	E = d[:,11]
	Ec= d[:,12]
	F = d[:,13]-3840 # CUDAマッチング時の都合上、320*12=3840 ピクセル数ほど 両ゼロピクセルが追加されているため

	fig = plt.figure(figsize=(20,12))	#figsize=（横幅、縦幅）

	#queryimgのピクセル数
	ax1 = fig.add_subplot(1,2,1)
	ax1.set_title("query=("+str(q[0])+","+str(q[1])+","+str(q[2])+"),("+str(q[3])+","+str(q[4])+")" ,size=12,color="black")
	ax1.plot(A, linestyle='solid',lw=0.1,  color ='grey', label="n(A)")
	plt.legend(loc="lower right")
	ax1.plot([Pbst,Pbst],[0,A.max()], linestyle='solid',lw=0.3, color ='red')

	#類似度グラフ
	ax11 = fig.add_subplot(1,2,2)#グラフを描画する場所を定義
	si=(Bc+C)/(A+B+C+D+E+0.0001) #類似度の評価方法を定義
	p=si.argmax()
	ax11.plot(si, linestyle='solid',lw=0.1,  color ='grey', label="si")
	plt.legend(loc="lower right")
	ax11.set_ylim(0,0.2)
	ax11.plot([Pbst,Pbst],[0,0.2], linestyle='solid',lw=0.3, color ='red')#真値座標を定義
	ax11.plot(p,si[p], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	ax11.text(0,0.15, str(p),size=10,color="blue") #推定位置座標
	ax11.text(0,0.1, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")",size=12,color="blue")
	ax11.text(0,0.0, "d:"+str(round(dist[p],4))+", n:"+str(round(n[p],3)),size=12,color="blue")

	r_file = "C:\\Users\\k0913\\Desktop\\EXP_&_DEMO\\EXP_fixedpoint_try\\result_image"

	fig.savefig(r_file + "\\" + "img.png")

	plt.show()#グラフを表示

	sonzai3.text(0,0.15, str(maxd),size=10,color="blue") #推定位置座標
	#sonzai.plot(dd, 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	sonzai3.plot(dd3, linestyle='solid',lw=0.1,  color ='blue', label="reliability")
	#fig.tight_layout()              #レイアウトの設定


	cwd = os.getcwd()	#current working directory

	#r_file = "C:\\Users\\k0913\\Desktop\\EXP_&_DEMO\\EXP_fixedpoint_miki1\\result_image"
	r_file = cwd + "\\result_image"

	print(qid)

	fig.savefig(r_file + "\\" + str(qid) +".png")

	plt.show()#グラフを表示


	plt.savefig(r_file + "\\" + str(qid) +".png")
def plot_value7(d, q, dist, n, def_ang, f_dir, fname, dd, qid, dd2, dd3, dd5):
	Pbst = n.argmin()#normの最小値
	A = d[:,6]
	B = d[:,7]
	Bc= d[:,8]
	C = d[:,9]
	D = d[:,10]
	E = d[:,11]
	Ec= d[:,12]
	F = d[:,13]-3840 # CUDAマッチング時の都合上、320*12=3840 ピクセル数ほど 両ゼロピクセルが追加されているため
	sim = d[:,14]
	dist = dd5[:,1]
	du = dd5[:,2]
	devi = dd5[:,3]
	hote = dd5[:,4]
	print(sim)

	plt.figure(figsize=(25,70))	#figsize=（横幅、縦幅）
	plt.subplots_adjust(wspace=0.5, hspace=0.5)
	plt.rcParams["font.size"] = 30

	#queryimgのピクセル数
	plt.subplot(5,4,1)
	plt.title("query pixel" ,color="black")
	plt.xlabel('DB imageID')
	plt.ylabel('number')
	plt.title("query=("+str(q[0])+","+str(q[1])+","+str(q[2])+"),("+str(q[3])+","+str(q[4])+")" ,color="black")
	plt.plot(A, linestyle='solid',lw=0.1,  color ='grey', label="query_pixel")
	plt.legend(loc="lower right")
	plt.plot([Pbst,Pbst],[0,A.max()], linestyle='solid',lw=0.3, color ='red')

	#類似度グラフ
	plt.subplot(5,4,2)#グラフを描画する場所を定義
	plt.title("UMap" ,color="black")
	plt.xlabel('DB imageID')
	plt.ylabel('similarity')
	si=(Bc+C)/(A+B+C+D+E+0.0001) #類似度の評価方法を定義
	norm = n


	dds = dict(zip(norm,dd))
	s_dds = sorted(dds.items(), key=lambda x:x[0])
	ddx = []
	ddy = []
	for x in range(len(s_dds)):
		ddx.append(s_dds[x][0])
	for y in range(len(s_dds)):
		ddy.append(s_dds[y][1])

	#類似度グラフの横軸を誤差が小さい順に並べ替える
	dsi = dict(zip(norm,sim))
	s_dsi = sorted(dsi.items(), key=lambda x:x[0])
	myx = []
	myy = []
	for x in range(len(s_dsi)):
		myx.append(s_dsi[x][0])
	for y in range(len(s_dsi)):
		myy.append(s_dsi[y][1])
	p=sim.argmax()
	maxd = dd.argmax()
	plt.plot(sim, linestyle='solid',lw=0.1,  color ='grey', label="simirality")
	plt.legend(loc="lower right")
	#plt.ylim(0,0.2)
	plt.plot([Pbst,Pbst],[0,0.2], linestyle='solid',lw=0.3, color ='red')#真値座標を定義
	#plt.plot([p,si[p]],[0,0.2], linestyle='solid',lw=0.3, color ='blue')#真値座標を定義
	plt.text(0,0.15, str(p),size=10,color="blue") #推定位置座標
	plt.text(0,0.1, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")",size=12,color="blue")

	#存在確率分布を作成
	maxd = sim.argmax()
	maxmyy = max(myy)
	minmyx = min(myx)
	plt.subplot(5,1,3)#グラフを描画する場所を定義
	plt.title("normal distribution" ,color="black")
	plt.xlabel('norm')
	plt.ylabel('similality')
	#print("myx: {}".format(myx))
	plt.plot(myx,myy, linestyle='solid',lw=0.1,  color ='grey', label="simirality")
	plt.scatter(myx, myy)
	#print(myy)
	#plt.plot(s_dsi, linestyle='solid',lw=0.1,  color ='red')
	#plt.plot([maxmyx,maxmyx], linestyle='solid',lw=0.1,  color ='red')
	plt.plot([maxmyy,maxmyy],[0,0.5], linestyle='solid',lw=0.7, color ='grey')#UMapとオドメトリを与えた推定位置を定義
	plt.plot(maxmyy, maxmyy, 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	plt.plot(ddx,ddy, linestyle='solid',lw=0.1,  color ='red', label="simirality")
	plt.text(0,0.15, str(maxd),size=10,color="blue") #推定位置座標

	#UMapを与えた存在確率分布
	plt.subplot(5,1,4)#グラフを描画する場所を定義
	plt.title("normal distribution and UMap" ,color="black")
	plt.xlabel('DB imageID')
	plt.ylabel('probability')
	maxd2 = dd2.argmax()
	plt.text(0,0.15, str(maxd2),size=10,color="blue") #推定位置座標
	#plt.plot(dd2, linestyle='solid',lw=0.1,  color ='red')
	plt.plot(ddx,ddy, linestyle='solid',lw=0.1,  color ='grey', label="simirality")
	plt.plot(dist, linestyle='solid',lw=0.1,  color ='grey', label="query_pixel")
	plt.plot(du, linestyle='solid',lw=0.1,  color ='black', label="query_pixel")
	plt.plot(devi, linestyle='solid',lw=0.1,  color ='red', label="query_pixel")
	plt.plot(hote, linestyle='solid',lw=0.1,  color ='blue', label="query_pixel")
	plt.plot(maxd2, dd2[maxd2], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	plt.plot([Pbst,Pbst],[0,1.0], linestyle='solid',lw=0.7, color ='red')#真値座標を定義
	plt.plot([maxd2,maxd2],[0,1.0], linestyle='solid',lw=0.7, color ='grey')#UMapとオドメトリを与えた推定位置を定義

	#偏差値を与えた存在確率分布を作成
	plt.subplot(5,1,5)#グラフを描画する場所を定義
	plt.title('reliability')
	plt.xlabel('DB imageID')
	plt.ylabel('realibility')
	maxd3 = dd3.argmax()
	plt.text(0,0.15, str(maxd3),color="blue") #推定位置座標

	plt.plot(dd3, linestyle='solid',lw=0.1,  color ='blue', label="reliability")
	plt.plot(dd, linestyle='solid',lw=0.1,  color ='red', label="reliability")
	plt.plot(sim, linestyle='solid',lw=0.1,  color ='grey', label="simirality")
	plt.plot(maxd3, dd3[maxd3], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='black', alpha=0.3)
	plt.plot([Pbst,Pbst],[0,1.0], linestyle='solid',lw=0.7, color ='red')#真値座標を定義
	plt.plot([maxd3,maxd3],[0,1.0], linestyle='solid',lw=0.7, color ='blue')#偏差値を与えた推定位置を定義
	plt.plot([maxd2,maxd2],[0,1.0], linestyle='solid',lw=0.7, color ='grey')#UMapとオドメトリを与えた推定位置を定義
	plt.plot([p,p],[0,1.0], linestyle='solid',lw=0.7, color ='black')#UMapの推定位置を定義
	#fig.tight_layout()              #レイアウトの設定

	cwd = os.getcwd()	#current working directory
	r_file = cwd + "\\result_image"
	plt.savefig(r_file + "\\" + str(qid) +".png")
	plt.show()#グラフを表示

def plot_value8(d, q, dist, n, def_ang, f_dir, fname, dd, qid, dd2, dd3, dd5):
	#出力値を定義
	Pbst = n.argmin()#normの最小値
	A = d[:,6]
	B = d[:,7]
	Bc= d[:,8]
	C = d[:,9]
	D = d[:,10]
	E = d[:,11]
	Ec= d[:,12]
	F = d[:,13]-3840 # CUDAマッチング時の都合上、320*12=3840 ピクセル数ほど 両ゼロピクセルが追加されているため
	sim = d[:,14] #UMap推定位置 blue
	dist = dd5[:,1] #存在確率分布 grey
	du = dd5[:,2] # 存在確率分布 + UMAp cyan
	devi = dd5[:,3] # 偏差値 green
	hote = dd5[:,4] # ホテリング理論 red

	#推定位置を出力
	maxsim = sim.argmax()
	maxd = dist.argmax()
	maxdu = du.argmax()
	maxdevi = devi.argmax()
	maxhote = hote.argmax()

	#----------------------------------------------------

	#横軸を真値との誤差(normにする)
	norm, s_sim, maxss, maxssi= normgraph(n, sim)
	norm, s_dist, maxsd,maxsdi = normgraph(n, dist)
	norm, s_du, maxsdu, maxsdui = normgraph(n, du)
	norm, s_devi, maxsdevi, maxsdevii = normgraph(n, devi)
	norm, s_hote, maxshote, maxshotei = normgraph(n, hote)
	print(len(sim))
	print(len(s_dist))
	print(len(norm))

	min_t, ti = min_value(n)
	#print(norm)


	#-----------------------------------------------------


	#--------------------------------------------------------
	#グラフを定義
	plt.figure(figsize=(90.0,90.0))	#figsize=（横幅、縦幅）
	plt.subplots_adjust(wspace=0.5, hspace=0.5)
	plt.rcParams["font.size"] = 30
	#-----------------------------------------------------------

	#queryimgのピクセル数
	plt.subplot(11,1,1)
	plt.title("query pixel" ,color="black")
	plt.xlabel('DB imageID')
	plt.ylabel('number')
	plt.title("query=("+str(q[0])+","+str(q[1])+","+str(q[2])+"),("+str(q[3])+","+str(q[4])+")" ,color="black")
	plt.plot(A, linestyle='solid',lw=0.1,  color ='grey', label="query_pixel")
	plt.legend(loc="lower right")
	plt.plot([Pbst,Pbst],[0,A.max()], linestyle='solid',lw=0.3, color ='red')
	#----------------------------------------------------------------------------

	#類似度グラフ
	plt.subplot(11,1,2)#グラフを描画する場所を定義
	plt.title("UMap" ,color="black")
	plt.xlabel('DB imageID')
	plt.ylabel('similarity')
	#si=(Bc+C)/(A+B+C+D+E+0.0001) #類似度の評価方法を定義
	#plt.plot(sim, linestyle='solid',lw=0.1,  color ='blue', label="simirality")
	plt.plot(s_sim, linestyle='solid',lw=0.2,  color ='powderblue', label="simirality")
	#plt.scatter(s_sim)
	plt.legend(loc="lower right")
	#plt.ylim(0,0.2)
	plt.plot([Pbst,Pbst],[0,0.2], linestyle='solid',lw=0.3, color ='gold')#真値座標を定義
	#plt.plot([maxsim,si[p]],[0,0.2], linestyle='solid',lw=0.3, color ='blue')#真値座標を定義
	plt.plot([maxsim,maxsim],[0,0.5], linestyle='solid',lw=0.7, color ='red')#偏差値を与えた推定位置を定義
	plt.plot([maxss,maxss],[0,0.5], linestyle='solid',lw=0.7, color ='blue')#偏差値を与えた推定位置を定義
	plt.text(0,0.15, str(maxsim),size=10,color="black") #推定位置座標
	plt.text(0,0.1, "pre-posi=("+str(d[maxsim,1])+","+str(d[maxsim,2])+","+str(d[maxsim,3])+"),("+str(d[maxsim,4])+","+str(d[maxsim,5])+")",size=12,color="blue")
	#--------------------------------------------------------------

	#存在確率分布を作成
	plt.subplot(11,1,3)#グラフを描画する場所を定義
	plt.title("normal distribution" ,color="black")
	plt.xlabel('norm')
	plt.ylabel('reliabiity')
	plt.xticks(np.arange(min(norm), max(norm)+1, 1.0))
	#plt.plot(norm, dist, linestyle='solid',lw=0.2,  color ='red', label="simirality")
	plt.plot(norm, s_dist, linestyle='solid',lw=0.01,  color ='lightsteelblue', label="simirality")
	#plt.scatter(norm, s_sim)
	#plt.plot(norm[maxsdi],norm[maxsdi], [0,0.5], linestyle='solid',lw=0.2, color ='grey')#UMapとオドメトリを与えた推定位置を定義
	plt.axvline(norm[maxsdi], ls = "--", color = "navy")
	plt.axvline(norm[ti], ls = "--", color = "gold")
	plt.plot(norm[maxsdi], maxsd, 'o', markersize=8, markeredgewidth=1.89, color='none', markeredgecolor='red', alpha=0.3)
	#plt.plot([Pbst,Pbst],[0,1.0], linestyle='solid',lw=0.7, color ='gold')#真値座標を定義
	plt.text(0,0.6, str(maxsd),size=10,color="blue") #推定位置座標
	#------------------------------------------------------------------

	#UMapを与えた存在確率分布
	plt.subplot(11,1,4)#グラフを描画する場所を定義
	plt.title("normal distribution and UMap" ,color="black")
	plt.xlabel('norm')
	plt.ylabel('probability')
	plt.text(0,0.6, str(maxdu),size=10,color="blue") #推定位置座標
	#plt.plot(dd2, linestyle='solid',lw=0.1,  color ='red')
	#plt.plot(ddx,ddy, linestyle='solid',lw=0.1,  color ='grey', label="simirality")
	#plt.plot(dist, linestyle='solid',lw=0.1,  color ='grey', label="query_pixel")
	plt.xticks(np.arange(min(norm), max(norm)+1, 1.0))
	plt.axvline(norm[maxsdui], ls = "--", color = "cyan")
	plt.axvline(norm[ti], ls = "--", color = "gold")
	plt.plot(norm[maxsdui], maxsd, 'o', markersize=8, markeredgewidth=1.89, color='none', markeredgecolor='red', alpha=0.3)
	#plt.plot(norm, du, linestyle='solid',lw=0.1,  color ='cyan', label="query_pixel")
	plt.plot(norm, s_du, linestyle='solid',lw=0.2,  color ='lightsteelblue', label="simirality")
	#plt.scatter(norm, s_du)
	#plt.plot(devi, linestyle='solid',lw=0.1,  color ='red', label="query_pixel")
	#plt.plot(hote, linestyle='solid',lw=0.1,  color ='blue', label="query_pixel")
	#plt.plot(maxdu, dd2[maxdu], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	#plt.plot([Pbst,Pbst],[0,1.0], linestyle='solid',lw=0.7, color ='gold')#真値座標を定義
	#plt.plot([maxdu,maxdu],[0,0.6], linestyle='solid',lw=0.7, color ='cyan')#UMapとオドメトリを与えた推定位置を定義
	#---------------------------------------------------------------------------------

	#偏差値を与えた存在確率分布を作成
	plt.subplot(11,1,5)#グラフを描画する場所を定義
	plt.title('deviation value')
	plt.xlabel('norm')
	plt.ylabel('probability of existence')
	plt.xticks(np.arange(min(norm), max(norm)+1, 1.0))
	plt.axvline(norm[maxsdevii], ls = "--", color = "green")
	plt.axvline(norm[ti], ls = "--", color = "gold")
	plt.plot(norm[maxsdevii], maxsd, 'o', markersize=8, markeredgewidth=1.89, color='none', markeredgecolor='red', alpha=0.3)
	#maxd3 = dd3.argmax()
	plt.text(0,0.1, str(maxdevi),color="blue") #推定位置座標
	plt.plot(norm, s_devi, linestyle='solid',lw=0.01,  color ='lightgreen', label="simirality")
	#plt.scatter(norm, s_devi)
	#plt.plot(dd3, linestyle='solid',lw=0.1,  color ='blue', label="reliability")
	#plt.plot(dd, linestyle='solid',lw=0.1,  color ='red', label="reliability")
	#plt.plot(sim, linestyle='solid',lw=0.1,  color ='grey', label="simirality")
	#plt.plot(maxdevi, dd3[maxdevi], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='black', alpha=0.3)
	#plt.plot([Pbst,Pbst],[0,0.6], linestyle='solid',lw=0.7, color ='gold')#真値座標を定義
	#plt.plot([maxdevi,maxdevi],[0,0.6], linestyle='solid',lw=0.7, color ='blue')#偏差値を与えた推定位置を定義
	#plt.plot([maxsim,maxsim],[0,0.6], linestyle='solid',lw=0.7, color ='black')#UMapの推定位置を定義
	#------------------------------------------------------------------

	#ホテリング理論を与えた存在確率分布を作成
	plt.subplot(11,1,6)#グラフを描画する場所を定義
	plt.title('Hotelling')
	plt.xlabel('norm')
	plt.ylabel('probability')
	plt.text(0,0.6, str(maxhote),color="blue") #推定位置座標
	plt.xticks(np.arange(min(norm), max(norm)+1, 1.0))
	plt.axvline(norm[maxshotei], ls = "--", color = "red")
	plt.axvline(norm[ti], ls = "--", color = "gold")
	plt.plot(norm[maxshotei], maxsd, 'o', markersize=8, markeredgewidth=1.89, color='none', markeredgecolor='red', alpha=0.3)
	plt.plot(norm, s_hote, linestyle='solid',lw=0.01,  color ='lavenderblush', label="simirality")
	#plt.scatter(norm, s_hote)
	#plt.plot(maxhote, dd3[maxhote], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='black', alpha=0.3)
	#plt.plot([Pbst,Pbst],[0,0.6], linestyle='solid',lw=0.7, color ='gold')#真値座標を定義
	#plt.plot([maxhote,maxhote],[0,0.6], linestyle='solid',lw=0.7, color ='blue')#偏差値を与えた推定位置を定義
	#plt.plot([maxsim,maxsim],[0,0.6], linestyle='solid',lw=0.7, color ='black')#UMapの推定位置を定義
	#-----------------------------------------------------

	#UMap + dist + du
	plt.subplot(11,1,7)#グラフを描画する場所を定義
	plt.title('UMap, distribution, UMapdist')
	plt.xlabel('norm')
	plt.ylabel('probability')
	plt.text(0,0.6, str(maxhote),color="blue") #推定位置座標
	plt.xticks(np.arange(min(norm), max(norm)+1, 1.0))
	plt.axvline(norm[maxssi], ls = "--", color = "blue")
	plt.axvline(norm[maxsdui], ls = "--", color = "black")
	plt.axvline(norm[maxsdi], ls = "--", color = "cyan")
	plt.axvline(norm[ti], ls = "--", color = "gold")
	plt.plot(norm[maxssi], maxsd, 'o', markersize=8, markeredgewidth=1.89, color='none', markeredgecolor='red', alpha=0.3)
	plt.plot(norm, s_sim, linestyle='solid',lw=0.01,  color ='powderblue', label="simirality")
	plt.plot(norm, s_dist, linestyle='solid',lw=0.01,  color ='grey', label="simirality")
	plt.plot(norm, s_du, linestyle='solid',lw=0.01,  color ='lavenderblush', label="simirality")
	#plt.plot(maxsd, dd3[maxsd], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='black', alpha=0.3)
	#plt.plot(maxsim, dd3[maxsim], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='black', alpha=0.3)
	# plt.plot([Pbst,Pbst],[0,0.6], linestyle='solid',lw=0.7, color ='gold')#真値座標を定義
	# plt.plot([maxsd,maxsd],[0,0.6], linestyle='solid',lw=0.7, color ='grey')#偏差値を与えた推定位置を定義
	# plt.plot([maxsdu,maxsdu],[0,0.6], linestyle='solid',lw=0.7, color ='cyan')#偏差値を与えた推定位置を定義
	# plt.plot([maxsim,maxsim],[0,0.6], linestyle='solid',lw=0.7, color ='blue')#UMapの推定位置を定義
	#-----------------------------------------------------

	#UMap + devi
	plt.subplot(11,1,8)#グラフを描画する場所を定義
	plt.title('UMap and deviation value')
	plt.xlabel('norm')
	plt.ylabel('probability')
	plt.text(0,0.6, str(maxdevi),color="blue") #推定位置座標
	plt.xticks(np.arange(min(norm), max(norm)+1, 1.0))
	plt.axvline(norm[maxssi], ls = "--", color = "blue")
	plt.axvline(norm[maxsdevii], ls = "--", color = "green")
	plt.axvline(norm[ti], ls = "--", color = "gold")
	plt.plot(norm[maxssi], maxsd, 'o', markersize=8, markeredgewidth=1.89, color='none', markeredgecolor='red', alpha=0.3)
	plt.plot(norm, s_sim, linestyle='solid',lw=0.01,  color ='powderblue', label="simirality")
	plt.plot(norm, s_devi, linestyle='solid',lw=0.01,  color ='lightgreen', label="simirality")
	#plt.plot(maxsdevi, dd3[maxsdevi], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='black', alpha=0.3)
	# plt.plot(maxsim, dd3[maxsim], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	# plt.plot([Pbst,Pbst],[0,0.6], linestyle='solid',lw=0.7, color ='gold')#真値座標を定義
	# plt.plot([maxsdevi,maxsdevi],[0,0.6], linestyle='solid',lw=0.7, color ='green')#偏差値を与えた推定位置を定義
	# plt.plot([maxsim,maxsim],[0,0.6], linestyle='solid',lw=0.7, color ='blue')#UMapの推定位置を定義
	#-----------------------------------------------------

	#UMap + hote
	plt.subplot(11,1, 9)#グラフを描画する場所を定義
	plt.title('UMap and Hotelling')
	plt.xlabel('norm')
	plt.ylabel('probability')
	plt.text(0,0.6, str(maxhote),color="blue") #推定位置座標
	plt.xticks(np.arange(min(norm), max(norm)+1, 1.0))
	plt.axvline(norm[maxssi], ls = "--", color = "blue")
	plt.axvline(norm[maxshotei], ls = "--", color = "green")
	plt.axvline(norm[ti], ls = "--", color = "gold")
	plt.plot(norm[maxssi], maxsd, 'o', markersize=8, markeredgewidth=1.89, color='none', markeredgecolor='red', alpha=0.3)
	plt.plot(norm, s_sim, linestyle='solid',lw=0.2,  color ='powderblue', label="simirality")
	plt.plot(norm, s_hote, linestyle='solid',lw=0.2,  color ='lavenderblush', label="simirality")
	#plt.plot(maxsd, dd3[maxshote], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='black', alpha=0.3)
	# plt.plot(maxsim, dd3[maxsim], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	# plt.plot([Pbst,Pbst],[0,0.6], linestyle='solid',lw=0.7, color ='gold')#真値座標を定義
	# plt.plot([maxshote,maxshote],[0,0.6], linestyle='solid',lw=0.7, color ='red')#偏差値を与えた推定位置を定義
	# plt.plot([maxsim,maxsim],[0,0.6], linestyle='solid',lw=0.7, color ='blue')#UMapの推定位置を定義
	#-----------------------------------------------------

	#UMap + dist + du
	plt.subplot(11,1,10)#グラフを描画する場所を定義
	plt.title('UMap and distribution and UMapdist')
	plt.xlabel('norm')
	plt.ylabel('probability')
	plt.text(0,0.6, str(maxhote),color="blue") #推定位置座標
	plt.plot(s_sim, linestyle='solid',lw=0.2,  color ='powderblue', label="simirality")
	plt.plot(s_dist, linestyle='solid',lw=0.2,  color ='grey', label="simirality")
	plt.plot(s_du, linestyle='solid',lw=0.2,  color ='cyan', label="simirality")
	#plt.plot(maxsd, dd3[maxsd], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='black', alpha=0.3)
	plt.plot(maxsim, dd3[maxsim], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	plt.plot([Pbst,Pbst],[0,0.6], linestyle='solid',lw=0.7, color ='gold')#真値座標を定義
	plt.plot([maxsd,maxsd],[0,0.6], linestyle='solid',lw=0.7, color ='grey')#偏差値を与えた推定位置を定義
	plt.plot([maxsdu,maxsdu],[0,0.6], linestyle='solid',lw=0.7, color ='cyan')#偏差値を与えた推定位置を定義
	plt.plot([maxsim,maxsim],[0,0.6], linestyle='solid',lw=0.7, color ='blue')#UMapの推定位置を定義
	#-----------------------------------------------------

	#UMap + dist + du
	plt.subplot(11,1,11)#グラフを描画する場所を定義
	plt.title('reliability')
	plt.xlabel('norm')
	plt.ylabel('realibilty')
	plt.text(0,0.6, str(maxhote),color="blue") #推定位置座標
	plt.plot(s_sim, linestyle='solid',lw=0.2,  color ='powderblue', label="simirality")
	plt.plot(s_dist, linestyle='solid',lw=0.2,  color ='grey', label="simirality")
	plt.plot(s_du, linestyle='solid',lw=0.2,  color ='cyan', label="simirality")
	#plt.plot(maxsd, dd3[maxsd], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='black', alpha=0.3)
	# plt.plot(maxsim, dd3[maxsim], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	# plt.plot([Pbst,Pbst],[0,0.6], linestyle='solid',lw=0.7, color ='gold')#真値座標を定義
	# plt.plot([maxsd,maxsd],[0,0.6], linestyle='solid',lw=0.7, color ='grey')#偏差値を与えた推定位置を定義
	# plt.plot([maxsdu,maxsdu],[0,0.6], linestyle='solid',lw=0.7, color ='cyan')#偏差値を与えた推定位置を定義
	# plt.plot([maxsim,maxsim],[0,0.6], linestyle='solid',lw=0.7, color ='blue')#UMapの推定位置を定義
	#-----------------------------------------------------
	#グラフを保存
	cwd = os.getcwd()	#current working directory
	r_file = cwd + "\\result_image"
	plt.savefig(r_file + "\\" + str(qid) +".png")
	plt.show()#グラフを表示

def best_th(d):		# それぞれの座標(x,y)で　sim_index が最大となる thを抜き出して data を作成
	R, C =d.shape
	print(R,C)

	d_out = np.empty((0,12))	# 出力用配列の準備

	vX = np.unique(d[:,1])
	vY = np.unique(d[:,2])

	for x in vX:
		for y in vY:
			d1=d[np.where(d[:,1]==x)]	# d から 1列目=x の行を抜き出し d1 とする
			d2=d1[np.where(d1[:,2]==y)]	# d1 から 2列目=y の行を抜き出し d2 とする
			d3=d2[np.argmax(d2[:,11]),:]	# d2 からsim_index が最大の行を抜き出し d3 とする
			d_out=np.vstack((d_out,d3))

	return d_out

def main():
		files_w_n, results_w_n, distribution_w_n = inputfile_listing()#csvファイルを選択
		end=0
		n=0
		qnum = 0

		#resultsファイルの選択（１つのcsvに１回の実験結果が保存されているため１回選択すればよい）
		for x in range(len(results_w_n[:,0])):
			print(results_w_n[x,0], ": ", results_w_n[x,1])


		print("Insert -1 to stop the process")
		print("Select the No.: ", end='')
		qn=int(input())

		#選択したresultのcsvファイル
		resultsname = results_w_n[qn,1]
		#print(resultsname)

		query_number = len(files_w_n)
		print("query_number : {}".format(query_number))
		print("qnum : {}".format(qnum))

	    #UMapマッチング結果のcsvファイルを選択（queryがごとのためquery枚数分繰り返す）
		while(query_number != qnum):
			n = qnum
			print(qnum)

			if (n>len(files_w_n[:,0])): break
			if (n<0): break
			#選択したqueryの結果csvファイル
			fullname = files_w_n[n,1]

			print(fullname)

			distname = distribution_w_n[n,1]
			print(distname)

			#p = fullname.find('EXP_fixedpoint_miki2')
			p = fullname.find(workspace)
			fname = fullname[p+30:-23]
			db_name = fullname[p+42:-4]
			f_dir = distname[p+41:-4]	#p_directory

			print("db_name : {}".format(db_name))
			print("fname : {}".format(fname))
			print("f_dir : {}".format(f_dir))


 	        #UMapマッチング結果をdataへ格納する
			data = read_txt(fullname)

	        #実験結果をres_dataへ格納する
			res_data = read_resulttxt(resultsname)
			print("resultsname : {}".format(resultsname))
			#print(res_data)
			query_id = res_data[qnum,0]
			#query_id = res_data[qnum]
			print (query_id)
	        #query = get_query_info(resultsname, query_id, res_data)

			#query画像の真値座標を取得
			x,y,z,th,ph,query = get_GT_position(resultsname, query_id, res_data, n)

			#存在確率分布のデータを取り込む
			dist_data = read_disttxt(distname)
			dist2_data = read_dist2txt(distname)
			dist3_data = read_dist3txt(distname)
			dist5_data = read_dist5txt(distname)
			dist, norm, def_ang= calc_distance(query,data)

			plot_value8(data, query, dist, norm, def_ang, f_dir, fname, dist_data, query_id, dist2_data, dist3_data, dist5_data)

			qnum = qnum + 1
			print(qnum)

if __name__ == "__main__":
	main()
	#main_d()
