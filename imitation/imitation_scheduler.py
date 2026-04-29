from __future__ import annotations
import random
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from simulator.models import Job, Cluster
from simulator.scheduler_interface import SchedulerInterface


def pair_features(job: Job, gpu, current_time: float):
    age = max(0.0, current_time - job.arrival_time)

    return np.array([
        job.gpu_mem_required / 80.0,
        job.gpu_util_intensity / 100.0,
        job.model_size / 80.0,
        job.batch_size / 64.0,
        job.seq_len / 4096.0,
        job.true_latency / 200.0,
        age / 1000.0,

        gpu.free_memory / 80.0,
        gpu.current_util / 100.0,
        gpu.num_running / 8.0,
    ], dtype=np.float32)


def expert_score(job, gpu, current_time,
                 beta=0.20,
                 gamma=0.15,
                 alpha=0.10):

    age = current_time - job.arrival_time

    runtime = job.true_latency
    fairness_bonus = beta * age
    fragmentation = gamma * abs(gpu.free_memory - job.gpu_mem_required)
    interference = alpha * gpu.current_util

    return runtime - fairness_bonus + fragmentation + interference


def generate_labels(jobs, cluster, current_time):
    X = []
    y = []

    for job in jobs:
        for gpu in cluster.gpus:
            if gpu.can_fit(job):
                X.append(pair_features(job, gpu, current_time))
                y.append(expert_score(job, gpu, current_time))

    return np.array(X), np.array(y)


class PairNet(nn.Module):
    def __init__(self, in_dim=10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(train_jobs,
                cluster,
                epochs=10,
                lr=1e-3):

    model = PairNet()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):

        X, y = generate_labels(train_jobs, cluster, current_time=0.0)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        pred = model(X)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"Epoch {epoch+1}/{epochs}  Loss={loss.item():.4f}")

    return model


# Scheduler
class ImitationScheduler(SchedulerInterface):

    def __init__(self, model):
        self.model = model

    @property
    def name(self):
        return "ImitationNN"

    def on_job_arrival(self, job, queue, cluster, current_time):
        return self._schedule(queue, cluster, current_time)

    def on_job_completion(self, completed_job, queue, cluster, current_time):
        return self._schedule(queue, cluster, current_time)

    def _schedule(self, queue, cluster, current_time):

        assignments = []
        used = set()

        while True:

            candidates = []

            for job in queue:
                if job.job_id in used:
                    continue

                for gpu in cluster.gpus:
                    if gpu.can_fit(job):

                        feat = pair_features(job, gpu, current_time)
                        x = torch.tensor(feat).float().unsqueeze(0)

                        with torch.no_grad():
                            score = self.model(x).item()

                        candidates.append((score, job, gpu))

            if not candidates:
                break

            candidates.sort(key=lambda t: t[0])   # lower better
            best_score, best_job, best_gpu = candidates[0]

            assignments.append((best_job, best_gpu.gpu_id))
            used.add(best_job.job_id)

            best_gpu.used_memory += best_job.gpu_mem_required

        for job, gpu_id in assignments:
            cluster.gpus[gpu_id].used_memory -= job.gpu_mem_required

        return assignments