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
				usecols=(0,1,2,3,4,5,6)
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
	hd = dd5[:,5] # ホテリング理論 red
	hu = dd5[:,6] # ホテリング理論 red



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
	norm, s_hd, maxshd, maxshdi = normgraph(n, hd)
	norm, s_hu, maxshu, maxshui = normgraph(n, hu)
	print(len(sim))
	print(len(s_dist))
	print(norm[0])

	min_t, ti = min_value(n)
	print("min_t : {}".format(min_t))
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
	plt.plot(norm, s_hd, linestyle='solid',lw=0.2,  color ='powderblue', label="simirality")
	plt.axvline(norm[ti], ls = "--", color = "gold")
	plt.axvline(norm[maxshdi], ls = "--", color = "green")
	#-----------------------------------------------------

	#UMap + dist + du
	plt.subplot(11,1,11)#グラフを描画する場所を定義
	plt.title('reliability')
	plt.xlabel('norm')
	plt.ylabel('realibilty')
	plt.text(0,0.6, str(maxhote),color="blue") #推定位置座標
	plt.plot(norm, s_hd, linestyle='solid',lw=0.2,  color ='powderblue', label="simirality")
	plt.plot(norm, s_hu, linestyle='solid',lw=0.2,  color ='lavenderblush', label="simirality")
	plt.axvline(norm[maxshdi], ls = "--", color = "red")
	plt.axvline(norm[maxshui], ls = "--", color = "blue")
	plt.axvline(norm[ti], ls = "--", color = "gold")
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
	#plt.show()#グラフを表示

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
