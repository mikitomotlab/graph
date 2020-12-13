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

workspace = 'EXP_fixedpoint_miki1'


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
					usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13)
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

def get_query_info(fullname):
	f=open(fullname)
	line = f.readline()
	line = f.readline()	# 2行目を読み込んで、line を上書き
	f.close
	line = "10" + line
	query = np.array(line.split(','), dtype = np.float)

	return query

#query画像の真値座標を定義
def get_GT_position(f_dir, query_id, res_data):
	#query画像ごとに真値座標は変化する必要がある
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

	#x, y, z, th, ph = res_data[query_id, 1], res_data[query_id, 2], 0.2, res_data[query_id, 3] , 0
	x, y, z, th, ph = res_data[1, 1], res_data[1, 2], 0.2, res_data[1, 3] , 0
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

	for i, dummy in enumerate(db[:,0]):
		#distance[i] = np.linalg.norm( db[i,1:4:1] - q[1:4:1] )

		a = np.array([db[i,1],db[i,2],db[i,3]])
		b = np.array([q[1],q[2],q[3]])
		distance[i] = np.linalg.norm( a - b )

		c = np.array((db[i,1],db[i,2],db[i,3], db[i,4]*np.pi/180, db[i,5]*np.pi/180))
		d = np.array([q[0],q[1],q[2],q[3]*np.pi/180,q[4]*np.pi/180])
		norm[i] = np.linalg.norm( c - d )

		e = np.array((db[i,4]*np.pi/180, db[i,5]*np.pi/180))
		f = np.array([q[3]*np.pi/180,q[4]*np.pi/180])
		def_ang[i] = np.linalg.norm( e - f )

	#print(q[1:4:1])
	return distance, norm, def_ang

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

def plot_values5(d, q, dist, n, def_ang, f_dir, fname, dd, qid):
	Pbst = n.argmin()#normの最小値
	A = d[:,6]
	B = d[:,7]
	Bc= d[:,8]
	C = d[:,9]
	D = d[:,10]
	E = d[:,11]
	Ec= d[:,12]
	F = d[:,13]-3840 # CUDAマッチング時の都合上、320*12=3840 ピクセル数ほど 両ゼロピクセルが追加されているため

	fig = plt.figure(figsize=(20,10))	#figsize=（横幅、縦幅）

	#queryimgのピクセル数
	ax1 = fig.add_subplot(2,2,1)
	ax1.set_title("query=("+str(q[0])+","+str(q[1])+","+str(q[2])+"),("+str(q[3])+","+str(q[4])+")" ,size=12,color="black")
	ax1.plot(A, linestyle='solid',lw=0.1,  color ='grey', label="query_pixel")
	plt.legend(loc="lower right")
	ax1.plot([Pbst,Pbst],[0,A.max()], linestyle='solid',lw=0.3, color ='red')

	#類似度グラフ
	ax11 = fig.add_subplot(2,2,2)#グラフを描画する場所を定義
	si=(Bc+C)/(A+B+C+D+E+0.0001) #類似度の評価方法を定義
	p=si.argmax()
	ax11.plot(si, linestyle='solid',lw=0.1,  color ='grey', label="simirality")
	plt.legend(loc="lower right")
	ax11.set_ylim(0,0.2)
	ax11.plot([Pbst,Pbst],[0,0.2], linestyle='solid',lw=0.3, color ='red')#真値座標を定義
	ax11.plot(p,si[p], 'o', markersize=8, markeredgewidth=0.2, color='none', markeredgecolor='blue', alpha=0.3)
	ax11.text(0,0.15, str(p),size=10,color="blue") #推定位置座標
	ax11.text(0,0.1, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")",size=12,color="blue")
	ax11.text(0,0.0, "d:"+str(round(dist[p],4))+", n:"+str(round(n[p],3)),size=12,color="blue")

	#存在確率分布を作成
	sonzai = fig.add_subplot(2,2,3)#グラフを描画する場所を定義
	sonzai.set_ylim(0,1000)
	sonzai.plot(dd, linestyle='solid',lw=0.1,  color ='grey', label="normal distribution")#plt.legend(loc="lower right")
	#sonzai.text(0,0.15, str(p),size=10,color="blue") #推定位置座標
	#sonzai.text(0,0.1, "pre-posi=("+str(d[p,1])+","+str(d[p,2])+","+str(d[p,3])+"),("+str(d[p,4])+","+str(d[p,5])+")",size=12,color="blue")
	#sonzai.text(0,10.0, "d:"+str(round(dist[p],4))+", n:"+str(round(n[p],3)),size=12,color="blue")

	#fig.tight_layout()              #レイアウトの設定

	r_file = "C:\\Users\\k0913\\Desktop\\EXP_&_DEMO\\EXP_fixedpoint_miki1\\result_image"

	print(qid)

	fig.savefig(r_file + "\\" + str(qid) +".png")

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
		qnum = 1

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
			cwd = os.getcwd()	#current working directory
			print("cwd : {}".format(cwd))

			#p = fullname.find('EXP_fixedpoint_miki1')
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
			query_id = res_data[qnum,0]
			print (query_id)
	        #query = get_query_info(resultsname, query_id, res_data)

			#query画像の真値座標を取得
			x,y,z,th,ph,query = get_GT_position(resultsname, query_id, res_data)

			#存在確率分布のデータを取り込む
			dist_data = read_disttxt(distname)
			dist, norm, def_ang = calc_distance(query,data)

			plot_values5(data, query, dist, norm, def_ang, f_dir, fname, dist_data, query_id)

			qnum = qnum + 1
			print(qnum)

if __name__ == "__main__":
	main()
	#main_d()
