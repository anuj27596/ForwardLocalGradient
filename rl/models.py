import torch
import torch.nn as nn
import torchvision


class LstmGate(nn.Module):
	def __init__(self, dim, activation):
		super().__init__()
		self.dim = dim
		self.activation = activation
		self.linear_x = nn.Linear(dim, dim)
		self.linear_h = nn.Linear(dim, dim)

	def forward(self, x, h):
		return self.activation(self.linear_x(x) + self.linear_h(h))


class LstmLayer(nn.Module):
	def __init__(self, batchsize, dim, **kwargs):
		super().__init__()
		self.batchsize = batchsize
		self.dim = dim
		self.h = torch.zeros(self.batchsize, dim).to('cuda')
		self.c = torch.zeros(self.batchsize, dim).to('cuda')
		self.input = LstmGate(dim, nn.Sigmoid())
		self.forget = LstmGate(dim, nn.Sigmoid())
		self.output = LstmGate(dim, nn.Sigmoid())
		self.cell = LstmGate(dim, nn.Tanh())

	def reset(self, term):
		self.h = self.h * ~term[:, None]
		self.c = self.c * ~term[:, None]

	def forward(self, inputs, recur_grad):
		if not recur_grad:
			self.h = self.h.detach()
			self.c = self.c.detach()
		i = self.input(inputs, self.h)
		f = self.forget(inputs, self.h)
		o = self.output(inputs, self.h)
		g = self.cell(inputs, self.h)
		self.c = f * self.c + i * g
		self.h = o * torch.tanh(self.c)
		return self.h


class RecurrerLSTM(nn.Module):
	def __init__(self, batchsize, dim = 512, num_layers = 3, **kwargs):
		super().__init__()
		self.layers = nn.ModuleList([
			LstmLayer(batchsize, dim)
			for _ in range(num_layers)
		])

	def reset(self, term):
		for layer in self.layers:
			layer.reset(term)

	def forward(self, inputs, recur_grad = True):
		x = inputs
		for layer in self.layers:
			x = layer(x, recur_grad)
		return x


class QLearner(nn.Module):
	def __init__(self, num_envs, **kwargs):
		super().__init__()
		# self.obs_shape = (3, 210, 160)
		self.num_actions = 4
		self.encoder = nn.Sequential(
			# *[*torchvision.models.resnet18().children()][:-1],
			nn.Conv2d(3, 32, 7, 3), nn.ReLU(),
			nn.Conv2d(32, 64, 3, 2), nn.ReLU(),
			nn.Conv2d(64, 128, 3, 2), nn.ReLU(),
			nn.Conv2d(128, 256, 3, 2), nn.ReLU(),
			nn.Conv2d(256, 512, 3, 2), nn.ReLU(),
			nn.AvgPool2d((3, 2)),
			nn.Flatten(),
		)
		self.recurrer = RecurrerLSTM(num_envs, num_layers = 1)
		self.qvalue = nn.Sequential(
			nn.Linear(512, 128), nn.ReLU(),
			nn.Linear(128, self.num_actions),
		)
		self.beta = 5.0

	def reset(self, term):
		self.recurrer.reset(term)

	def sample_action(self, values):
		policy = torch.softmax(self.beta * values, dim = 1)
		action = torch.multinomial(policy, num_samples = 1)[:, 0]
		return action

	def forward(self, obs, recur_grad = True):
		latents = self.encoder(obs.permute(0, 3, 1, 2))
		state = self.recurrer(latents, recur_grad)
		values = self.qvalue(state)
		action = self.sample_action(values)
		return action, values
