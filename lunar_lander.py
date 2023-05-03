import gym
import numpy as np
import cv2
import logging
import cgp


class CGPGymWrapper:
    """
    Gym wrapper for CGP. It contains objective function, envirnoment and render/log functions.
    """
    def __init__(self, env: gym.Env):
        self.env = env
        self.max_steps = 350

    def render(self, individual: cgp.IndividualSingleGenome, save: bool = False, save_path: str = ""):
        state, _ = self.env.reset()
        f = individual.to_numpy()
        video = []
        for _ in range(self.max_steps):
            action = f(state)
            action = np.clip(action, -1, 1)
            state, reward, done, truncation, info = self.env.step(action)
            img = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
            if save:
                video.append(img)
            cv2.imshow("test", img)
            cv2.waitKey(10)


        if save:
            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"MP4V"), 40, img.shape[:2][::-1])
            for frame in video:
                out.write(frame)
            out.release()

    def objective(self, individual: cgp.individual.IndividualBase):
        fitness = 0.0
        f = individual.to_numpy()
        for i in range(3):
            state, _ = self.env.reset()
            for _ in range(self.max_steps):
                action = f(state)
                action = np.clip(action, -1, 1)
                state, reward, done, truncated, info = self.env.step(action)
                fitness += reward
                if done or truncated:
                    break
        total_func_count = sum(individual.calculate_count_per_function().values())
        total_func_count = np.clip(total_func_count, 0, 12)
        ratio = pow(((total_func_count+1)/13), 3/5)
        ratio = 1/ratio if fitness < 0 else ratio
        individual.fitness = fitness/3
        individual.shitness = individual.fitness
        individual.fitness *= ratio

        return individual

    def log_end(self, logger: logging.Logger, population: cgp.Population):
        champion = population.champion
        logger.info(champion.calculate_count_per_function())


if __name__ == "__main__":
    pass