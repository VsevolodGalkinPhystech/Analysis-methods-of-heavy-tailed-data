#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 02:34:01 2020

@author: vsevolod
"""

import numpy as np
import matplotlib.pyplot as plt

import h5py as h5
import os.path as fs
import sys

from numba import njit, prange
from numba import float32, int32, float64, int64 
#%%
def saveHDF5file(PathToSave, SavedFileName, list_group_name, data):
  """
  Сохраение hdf5 файла
  """
  num_group = len(list_group_name)
  num_data = len(data)
  if num_group != num_data:
   raise RuntimeError('Список имен групп и длина списка данных не соответствуют!')
  
  ff = h5.File(fs.join(PathToSave, SavedFileName), 'w')
  for i, group in enumerate(list_group_name):
    ff.create_dataset(group, data = data[i])
  ff.close()
  return None

def readHDF5file(PathToSave, SavedFileName, list_group_name):
  """
  Загрузка hdf5 файла
  """
  data = []
  ff = h5.File(fs.join(PathToSave, SavedFileName), 'r')
  for group in list_group_name:
    data.append(ff[group][...])
  ff.close()
  return data
#%%
def FrechetDistribution(n, gamma):
  """
  Создание выборки из распределения Фреше с параметрами:
    n -- размер выборки
    gamma -- параметр распределения
  """
  samples = np.empty((n, ), dtype = np.float32)
  unifrom = np.random.uniform(0., 1.0, size = samples.shape)
  samples = (1./gamma)*((-1.*np.log(unifrom))**(-1.*gamma))
  return samples

def RVD(n, C, gamma):
  """
  Создание выборки из распределения с регулярно меняющимися хвостами
  F(x) = x^(-1/gamma) l(x) с параметрами:
    l(x) = const
    gamma
    n -- объем выборки
  """
  eps = 10**(-4)
  samples = np.empty((n, ), dtype = np.float32)
  unifrom = np.random.uniform(0., 1.0 - eps, size = samples.shape)
  samples = (C/(1. - unifrom))**(gamma)
  return samples

def Weibull(n, C, gamma):
  """
  Создание выборки из распределения Вейбула с параметрами:
    n -- объем выборки
    C, gamma -- константные параметры распределения
  """
  eps = 10**(-4)
  samples = np.empty((n, ), dtype = np.float32)
  unifrom = np.random.uniform(0., 1.0 - eps, size = samples.shape)
  samples = ((-1./C)*np.log(1. - unifrom))**(gamma)
  return samples

def PlotDistribution(samples, title = '', bins = 10):
  """
  Построение выборки в частотной оси -- статистический аналог функции плотности
  """
  fig = plt.figure(figsize = (10, 10,))
  plt.hist(samples, bins = bins, color = 'blue')
  plt.xlabel('x')
  plt.ylabel('frequency')
  plt.title(title)
  plt.show()
  return None
#%%
def CalculatrStatistic(samples, p):
  """
  Вычисление статистики ratio of the maximum to the sum, задание 2.
  """
  if p < 0:
    print('Error. p must be a positive number')
    sys.exit()
  Mn = np.max(np.abs(samples)**p)
  Sn = np.sum(np.abs(samples)**p)
  Rn = Mn/Sn
  return Rn

def CalculateDependenceStatP(samples_source, num_grid, p):
  """
  Вычисление завимисости статистики для задания 2, чтобы построить график 
  разброса. 
    sample_source -- статистическая выборка
    num_grid -- число узлов по параметру объема выборки
    p -- значение параметра статистики (аналог момента)
  """
  n_grid = np.linspace(50, samples_source.shape[0], num = num_grid, dtype = np.int32)
  dependence = np.empty(n_grid.shape , dtype = np.float32)
  samples_union = samples_source
  for i in range(len(n_grid)):
    samples = samples_union[:n_grid[i]]
    dependence[i] = CalculatrStatistic(samples, p)
  return dependence, n_grid

def PlotDependenceStatP(samples, p_list, num_grid=500):
  """
  Построение графика зависимости статистики (см. задание 2) от объема выборки 
  при различных значениях параметра p.
  """
  fig = plt.figure(figsize = (16, 10))
  for p in p_list:
    dep, n_grid = CalculateDependenceStatP(samples,\
                                       num_grid=num_grid, p = p)
    plt.plot(n_grid, dep, label = 'p = %d'%p)
  plt.xlabel('m')
  plt.ylabel(r'$R_n(p)$')
  plt.title(r'dependence $R_n(p)$ against $n$ for different $p$')
  plt.grid(True)
  plt.legend()
  plt.show()
  return None
#%%
def empiricalMean(samples, u):
  """
  Вычисление эмпирического среднего
  """
  ind = np.argwhere(samples > u)[:, 0]
  e_nu = 0.
  for i in ind:
    e_nu = e_nu + (samples[i] - u)
  e_nu = e_nu/len(ind)
  return e_nu

def plotE_nu(samples, u_min, u_max, n_u):
  """
  Построение графика зависимости эмпирического среднего от параметра u
  """
  u_array = np.linspace(u_min, u_max, num = n_u)
  e_nu_array = np.empty((n_u, ), dtype = np.float32)
  for i in range(n_u):
    e_nu_array[i] = empiricalMean(samples, u_array[i])
  
  plt.plot(u_array, e_nu_array, color = 'b')
  plt.scatter(u_array, e_nu_array, color = 'b', s = 15, alpha = 0.7)
  plt.xlabel('u')
  plt.ylabel(r'$e_n(u)$')
  plt.title('dependence $e_n(u)$ against $n$ for different $u$')
  plt.grid(True)
  plt.show()
  None  
#%%
@njit([float64[:](float64[:])])
def variation_series(samples):
  """
  Сортировка выборки и создание вариационного ряда
  """
  variation_series = np.sort(samples)
  return variation_series

@njit([float64(float64[:], int64)])
def Hill_estimator(var_samples, k):
  """
  Вычисление оценки Хилла
  """
  estimator = 0.
  for i in range(k):
    estimator += np.log(var_samples[-(i + 1)]) - np.log(var_samples[-k])
  estimator /= k+1
  return estimator

def plotHillEstimator(samples):
  """
  Построение оценки Хилла
  """
  n = len(samples)
  var_samples = variation_series(samples)
  k_array = np.arange(n - 1)
  hill_estim = np.empty(k_array.shape, dtype = np.float32)
  for k, k_iter in enumerate(k_array):
    hill_estim[k] = Hill_estimator(var_samples, k_iter)
    
  print(hill_estim[:20])
  plt.plot(k_array, hill_estim, color = 'b')
  plt.scatter(k_array, hill_estim, color = 'b', s = 15, alpha = 0.7)
  plt.xlabel('k')
  plt.ylabel('Hill estimator')
  plt.title('dependence Hill estimator for different k')
  plt.grid(True)
  plt.show()
  return None

def Ratio_estimator(var_samples, k):
  """
  Вычисление оценки Ratio
  """
  estimator = 0.
  x_trashhold = var_samples[-k]
  ind = np.argwhere(var_samples >= x_trashhold)[:, 0]
  for idx in ind:
    estimator += np.log(var_samples[idx]/x_trashhold)
  estimator /= len(ind)
  return estimator

def plotRatioEstimator(samples):
  """
  Построение оценки Ratio
  """
  n = len(samples)
  var_samples = variation_series(samples)
  x_trashold = np.asarray([i for i in range(1, n - 1)])
  ratio_estim = np.empty(x_trashold.shape, dtype = np.float32)
  for td_iter, td in enumerate(x_trashold):
    ratio_estim[td_iter] = Ratio_estimator(var_samples, td)
    
  plt.plot(x_trashold, ratio_estim, color = 'b')
  plt.scatter(x_trashold, ratio_estim, color = 'b', s = 15, alpha = 0.7)
  plt.xlabel('xn')
  plt.ylabel('Ratio estimator')
  plt.title('dependence Ratio  estimator for different xn')
  plt.grid(True)
  plt.show()
  return None

def Snk_value(var_samples, k):
  """
  Вычисление величины Snk для оценки moment
  """
  Snk = 0.
  for i in range(k):
    Snk+= (np.log(var_samples[-(i - 1)]) - np.log(var_samples[-k]))**2
  Snk /= k+1
  return Snk

def momentEstimator(samples, k):
  """
  Вычисление самой оценки moment
  """
  var_samples = variation_series(samples)
  estimator = 0.
  Snk = Snk_value(var_samples, k)
  hill_est = Hill_estimator(var_samples, k)
  estimator = hill_est + 1 -\
    0.5*((1 - ((hill_est)**2/Snk))**(-1))
  return estimator


def plotMomentEstimator(samples):
  """
  Построение оценки moment
  """
  n = len(samples)
  k_array = np.arange(n - 2)
  moment_estim = np.empty(k_array.shape, dtype = np.float32)
  for k, k_iter in enumerate(k_array):
    moment_estim[k] = momentEstimator(samples, k_iter)
    
  plt.plot(k_array, moment_estim, color = 'b')
  plt.scatter(k_array, moment_estim, color = 'b', s = 15, alpha = 0.7)
  plt.xlabel('k')
  plt.ylabel('Moment estimator')
  plt.title('dependence Moment estimator for different k')
  plt.grid(True)
  plt.show()
  return None


@njit([float64(float64[:], int64)])
def UH_i(var_samples, i_iter):
  """
  Вычисление величины UH_i для UH оценки
  """
  UH = 0.
  UH = var_samples[-i_iter]*Hill_estimator(var_samples, i_iter)
  return UH

@njit([float64(float64[:], int64)])
def UHEstimator(samples, k):
  """
  Вычисление UH оценки.
  """
  var_samples = variation_series(samples)
  estimator = 0.
  for i_iter in range(1, k):
    estimator += np.log(UH_i(var_samples, i_iter) + 0.01) - np.log(UH_i(var_samples, k + 1) + 0.01)
  
  estimator /= k + 1
  return estimator

def plotUHEstimator(samples):
  """
  Построение UH оценки.
  """
  n = len(samples)
  k_array = np.arange(2, n - 1)
  UH_estim = np.empty(k_array.shape, dtype = np.float32)
  for k, k_iter in enumerate(k_array):
    UH_estim[k] = UHEstimator(samples, k_iter)
  
  plt.plot(k_array, UH_estim, color = 'b')
  plt.scatter(k_array, UH_estim, color = 'b', s = 15, alpha = 0.7)
  plt.xlabel('k')
  plt.ylabel('UH estimator')
  plt.title('dependence UH estimator for different k')
  plt.grid(True)
  plt.show()
  return None


def PicandsEstimator(samples, k):
  """
  Вычисление оценки Picands
  """
  n = len(samples)
  if k > n/4.:
    raise Exception("wrong!")
  var_samples = variation_series(samples)
  S = (var_samples[-(k + 1)] - var_samples[-(2*k + 1)])\
    /(var_samples[-(2*k + 1)] - var_samples[-(4*k + 1)])
  estimator = 0.
  estimator = (1./ np.log(2))*np.log(S)
  return estimator

def plotPicandsEstimator(samples):
  """
  Построение оценки Picands
  """
  n = len(samples)
  k_array = np.arange(n // 4)
  Pic_estim = np.empty(k_array.shape, dtype = np.float32)
  for k, k_iter in enumerate(k_array):
    Pic_estim[k] = PicandsEstimator(samples, k_iter)
  plt.plot(k_array, Pic_estim, color = 'b')
  plt.scatter(k_array, Pic_estim, color = 'b', s = 15, alpha = 0.7)
  plt.xlabel('k')
  plt.ylabel('Pickands estimator')
  plt.title('dependence Pickands estimator for different k')
  plt.grid(True)
  plt.show()
  return None
#%%
def FrecheInverse(arg, gamma):
  """
  Обратная функция к функции распределения Фреше
  """
  return (1./gamma)*((-1.*np.log(arg))**(-1.*gamma))

def QQPlot(samples, gamma, ext):
  """
  Построение QQ графика
  """
  n = len(samples)
  var_samples = variation_series(samples)
  #var_samples = var_samples[:n - ext]
  var_samples = var_samples[::-1]
  F_inv = np.empty(var_samples.shape, dtype = np.float32)
  n = len(var_samples)
  for i in range(n):
    arg = (n - (i + 1) + 1)/(n + 1)
    F_inv[i] = FrecheInverse(arg, gamma)
  plt.plot(np.array([0, np.max(F_inv[ext:])]), np.array([0, np.max(F_inv[ext:])]),\
           color = 'red')
  plt.scatter(var_samples[ext:], F_inv[ext:], color = 'b', s = 15, alpha = 0.7)
  plt.xlabel(r'X_k')
  plt.ylabel('F inverse')
  plt.title('QQ plot')
  plt.grid(True)
  plt.show()
  return None

#%%
def grouping(samples, num_group):
  """
  Случайное разбиение выборки на группы
  """
  n = len(samples)
  if n < num_group:
    raise Exception("wrong!")
  perm = np.random.permutation(n)
  parts = np.array_split(samples[perm], num_group)
  return parts

def groupEstimateTailIndex(groups):
  """
  Вычисление групповой оценки хвостового индекса
  """
  n = len(groups)
  kl = np.empty((n, ), dtype = np.float32)
  for i in range(n):
   var_samples_on_group = variation_series(groups[i])
   M1, M2 = var_samples_on_group[-1], var_samples_on_group[-2]
   kl[i] = M2/M1
  zl = (1./n)*np.sum(kl)
  group_estimator = (1./zl) - 1.
  return group_estimator

def GEConfidenceInterval(groups):
  """
  Вычисление доверительного интервала групповой оценки хвостового индекса
  """
  n = len(groups)
  kl = np.empty((n, ), dtype = np.float32)
  for i in range(n):
   var_samples_on_group = variation_series(groups[i])
   M1, M2 = var_samples_on_group[-1], var_samples_on_group[-2]
   kl[i] = M2/M1
  conf_interval = np.empty((2, ), dtype = np.float32)
  k_ol = (1./n)*np.sum(kl)
  A = np.sum(kl**2) - k_ol
  lst = [(k_ol - ((1.96*np.sqrt(A)))/n)**(-1) - 1,\
         (k_ol - ((-1.96*np.sqrt(A)))/n)**(-1) - 1]
  conf_interval[0] = min(lst)
  conf_interval[1] = max(lst)
  return conf_interval
  
  """
  Построение групповой оценкци и доверительного интервала
  """
  n = len(samples)
  m_grid = np.linspace(10, n//2, num = n//2, dtype = np.int32)
  new_m_grid = []
  l_grid = []
  for i in range(len(m_grid)):
    if n % m_grid[i] == 0:
      new_m_grid.append(m_grid[i])
      l_grid.append(n//m_grid[i])
  l_grid, m_grid = np.asarray(l_grid), np.asarray(new_m_grid)
  group_estimators = np.empty(l_grid.shape,\
                              dtype = np.float32)
  CI = np.empty(l_grid.shape + (2, ),\
                              dtype = np.float32)
  for i in range(len(l_grid)):
    groups_tmp = grouping(samples, l_grid[i])
    group_estimators[i] = groupEstimateTailIndex(groups_tmp)
    CI[i] = GEConfidenceInterval(groups_tmp)
  plt.plot(m_grid, group_estimators,\
           color = 'b')
  plt.scatter(m_grid, group_estimators,\
              color = 'b', s = 15, alpha = 0.7)
  plt.fill_between(m_grid, CI[:, 0], CI[:, 1], color = 'blue', alpha = 0.7)
  plt.xlabel(r'm')
  plt.ylabel('Estimator')
  plt.title('Group estimaror of the tail index, n = %d'%(n))
  plt.grid(True)
  plt.show()
  return None
#%%
num_sample = 1000 # количество сэмплов выборки
gamma = 1.5 # параметр распределения Фреше
FrechetSamples = FrechetDistribution(num_sample, gamma) # геренация выборки
PlotDistribution(FrechetSamples, bins = 50) # построение выборки

plotE_nu(FrechetSamples, 0, 1000, 500) # построение эмпирического среднего

plotHillEstimator(FrechetSamples) # оценка Хилла
plotRatioEstimator(FrechetSamples) # оценка Ratio
plotMomentEstimator(FrechetSamples) # оценка Moment
plotUHEstimator(FrechetSamples) # UH оценка
plotPicandsEstimator(FrechetSamples) # Picands оценка

p_list = [1, 2, 3, 4]
PlotDependenceStatP(FrechetSamples, p_list) # построение ratio of the maximum to the sum

ext = 500 # объем исключенных экстремальных сэмплов
QQPlot(FrechetSamples, gamma, ext) # построение QQ plot
#%%
RVD_samples = RVD(500, 2., 0.5) # распределение с регулярно мен. хвостами
PlotDistribution(RVD_samples, 'RVD, l(x) = 1, gamma = 0.5', bins = 50) # построение

Weibull_samples = Weibull(500, 2., 3.) # распределение Вейбула
PlotDistribution(Weibull_samples, 'Weibull, c = 2, gamma = 3', bins = 50) # построение

plotHillEstimator(Weibull_samples) # построение оценки Хилла
#%%
num_sample = 1000 # количество сэмплов выборки
gamma = 1.5 # параметр распределения Фреше
FrechetSamples = FrechetDistribution(num_sample, gamma) # геренация выборки

plotGroupEstimator(FrechetSamples) # построение групповой оценки
