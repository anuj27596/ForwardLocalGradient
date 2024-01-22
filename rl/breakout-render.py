import gymnasium as gym

import matplotlib.pyplot as plt
import matplotlib.animation as animation


if __name__ == '__main__':
	
	env = gym.make("ALE/Breakout-v5")

	obs, info = env.reset()

	fig = plt.figure(figsize = (8, 8))
	frames = [[plt.imshow(obs, animated = True)]]

	for i in range(1000):
		action = env.action_space.sample()
		obs, rew, term, trunc, info = env.step(action)

		frames.append([plt.imshow(obs, animated = True)])

		if term or trunc:
			obs, info = env.reset()
			frames.append([plt.imshow(obs, animated = True)])

	env.close()

	ani = animation.ArtistAnimation(fig, frames, interval = 33, blit = True, repeat_delay = 200)
	plt.show()
