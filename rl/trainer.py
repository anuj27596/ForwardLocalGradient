import numpy as np

import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD

import tensorflow as tf

import gymnasium as gym

import matplotlib.pyplot as plt

from rl.models import QLearner

from tqdm import tqdm, trange


GAMMA = 0.99
NUM_ENV = 32


def dualize(module):
	setattr(module, 'param_stash', {})
	for name, param in list(module.named_parameters(recurse = False)):
		module.param_stash[name] = param
		delattr(module, name)
		setattr(module, name, fwAD.make_dual(param.detach().clone(), param.grad.clone()))


def undualize(module):
	for name, param in module.param_stash.items():
		delattr(module, name)
		module.register_parameter(name, param)
	delattr(module, 'param_stash')



if __name__ == '__main__':

	device = 'cuda'
	torch.autograd.set_detect_anomaly(True)

	env = gym.make_vec("ALE/Breakout-v5", num_envs = NUM_ENV)

	model = QLearner(NUM_ENV).to(device)

	optimizer = torch.optim.SGD(model.parameters(), lr = 5e-3, momentum = 0.9, weight_decay = 5e-4)

	n_epochs = 10000

	counts = []
	total_rewards = []
	losses = []

	for ep in trange(n_epochs):

		# compute guess

		optimizer.zero_grad()
		obs, info = env.reset()
		prev_value = None
		guess = [0.0 for _ in model.parameters()]

		for _ in range(8):
			obs = torch.Tensor(obs).to(device) / 255
			action, values = model(obs, recur_grad = False)
			
			if prev_value is not None:
				td_target = rew + GAMMA * values.detach().max(dim = 1).values
				loss = ((prev_value - td_target) ** 2).sum()
				loss.backward()
				guess = [p.grad + g for p, g in zip(model.parameters(), guess)]

			obs, rew, term, trunc, info = env.step(action)
			prev_value = values[range(NUM_ENV), action]
			rew = torch.tensor(rew).to(device)
			term = torch.tensor(term).to(device)

			if term.any().item():
				model.reset(term)


		# compute jvp & update

		sq_norm = sum([(g ** 2).sum() for g in guess])
		for p, g in zip(model.parameters(), guess):
			p.grad = g / sq_norm

		totrew = 0.0
		count = 0
		prev_value_list = []

		with torch.no_grad(), fwAD.dual_level():
			model.apply(dualize)

			obs, info = env.reset()
			prev_value = None
			loss = 0.0

			for step in range(256):
				obs = torch.Tensor(obs).to(device) / 255
				action, values = model(obs, recur_grad = False)
				
				# if prev_value is not None:
				# 	td_target = rew + GAMMA * values.detach().max(dim = 1).values
				# 	loss = loss + ((prev_value - td_target) ** 2).sum()

				obs, rew, term, trunc, info = env.step(action)
				totrew += rew.sum()

				# prev_value = values[range(NUM_ENV), action]
				rew = torch.tensor(rew).to(device)
				term = torch.tensor(term).to(device)

				prev_value_list.append(values[range(NUM_ENV), action])
				for i, pv in enumerate(prev_value_list):
					disc = GAMMA ** (step - i)
					prev_value_list[i] = pv - disc * rew

				if term.any().item():
					model.reset(term)
					count += int(term.sum().item())

			loss = sum([(pv ** 2).sum() for pv in prev_value_list])

			counts.append(count)
			total_rewards.append(totrew)
			losses.append(loss.detach().item())

			jvp = fwAD.unpack_dual(loss).tangent
			model.apply(undualize)

		for p in model.parameters():
			p.grad *= jvp


		optimizer.step()

		if (ep + 1) % 100 == 0:
			with tf.io.gfile.GFile('gs://ue-usrl-anuj/fwg-rl/2/counts.npy', 'w') as f:
				np.save(f, counts)
			with tf.io.gfile.GFile('gs://ue-usrl-anuj/fwg-rl/2/total_rewards.npy', 'w') as f:
				np.save(f, total_rewards)
			with tf.io.gfile.GFile('gs://ue-usrl-anuj/fwg-rl/2/losses.npy', 'w') as f:
				np.save(f, losses)
