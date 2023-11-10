RPTO_CONFIG = {
    "InvertedPendulum": {
        "num_epoch": 40,
        "step_per_epoch": 250,
        "real_step_per_collect": 1,
        "num_train_repeat": 10,
        "model_train_freq": 250,
        "model_retain_epochs": 1,
        "rollout_batch_size": int(100e3),
        "rollout_min_epoch": 1,
        "rollout_max_epoch": 15,
        "rollout_min_length": 1,
        "rollout_max_length": 1,
        "init_exploration_steps": 500,
    },
    "InvertedDoublePendulum": {
        "num_epoch": 80,
        "step_per_epoch": 250,
        "real_step_per_collect": 1,
        "num_train_repeat": 10,
        "dynamics_update_epoch": 2,
        "model_train_freq": 100,
        "model_retain_epochs": 1,
        "rollout_batch_size": int(100e3),
        "rollout_min_epoch": 1,
        "rollout_max_epoch": 15,
        "rollout_min_length": 1,
        "rollout_max_length": 1,
        "init_exploration_steps": 500,
    },
    "Swimmer": {
        "num_epoch": 80,
        "step_per_epoch": 250,
        "real_step_per_collect": 1,
        "num_train_repeat": 20,
        "dynamics_update_epoch": 5,
        "model_train_freq": 100,
        "model_retain_epochs": 1,
        "rollout_batch_size": int(100e3),
        "rollout_min_epoch": 1,
        "rollout_max_epoch": 15,
        "rollout_min_length": 1,
        "rollout_max_length": 1,
        "init_exploration_steps": 500,
        "policy_path": "/apdcephfs/share_1563664/ztjiaweixu/mbpo_sz/2023062711/Swimmer-v0",
    },
    "HalfCheetah": {
        "num_epoch": 100,
        "step_per_epoch": 1000,
        "real_step_per_collect": 1,
        "num_train_repeat": 1,
        "dynamics_update_epoch": 2,
        "model_train_freq": 250,
        "model_retain_epochs": 1,
        "rollout_batch_size": int(100e3),
        "rollout_min_epoch": 20,
        "rollout_max_epoch": 150,
        "rollout_min_length": 1,
        "rollout_max_length": 1,
        "init_exploration_steps": 5000,
        "policy_path": "/apdcephfs/share_1563664/ztjiaweixu/mbpo_sz/2023022301/HalfCheetah-v2",
    },
    "Hopper": {
        "num_epoch": 100,
        "step_per_epoch": 1000,
        "real_step_per_collect": 1,
        "num_train_repeat": 2,
        "dynamics_update_epoch": 5,
        "model_train_freq": 500,
        "model_retain_epochs": 1,
        "rollout_batch_size": int(100e3),
        "rollout_min_epoch": 20,
        "rollout_max_epoch": 150,
        "rollout_min_length": 1,
        "rollout_max_length": 1,
        "init_exploration_steps": 5000,
        "policy_path": "/apdcephfs/share_1563664/ztjiaweixu/mbpo_sz/2023060200/Hopper-v0",
    },
    "Walker2d": {
        "num_epoch": 100,
        "step_per_epoch": 1000,
        "real_step_per_collect": 1,
        "num_train_repeat": 5,
        "dynamics_update_epoch": 1,
        "model_train_freq": 250,
        "model_retain_epochs": 1,
        "rollout_batch_size": int(100e3),
        "rollout_min_epoch": 20,
        "rollout_max_epoch": 150,
        "rollout_min_length": 1,
        "rollout_max_length": 1,
        "init_exploration_steps": 5000,
        "policy_path": "/apdcephfs/share_1563664/ztjiaweixu/mbpo_sz/2023022301/Walker2d-v2",
    },
    "AntTruncated": {
        "num_epoch": 100,
        "step_per_epoch": 1000,
        "real_step_per_collect": 1,
        "num_train_repeat": 5,
        "dynamics_update_epoch": 2,
        "model_train_freq": 250,
        "model_retain_epochs": 1,
        "rollout_batch_size": int(100e3),
        "rollout_min_epoch": 20,
        "rollout_max_epoch": 100,
        "rollout_min_length": 1,
        "rollout_max_length": 1,
        "init_exploration_steps": 5000,
        "policy_path": "/apdcephfs/share_1563664/ztjiaweixu/mbpo_sz/2023022301/AntTruncated-v2",
    },
    "ContinuousAcrobot": {
        "num_epoch": 20,
        "step_per_epoch": 1000,
        "real_step_per_collect": 1,
        "num_train_repeat": 2,
        "dynamics_update_epoch": 5,
        "model_train_freq": 100,
        "model_retain_epochs": 1,
        "rollout_batch_size": int(100e3),
        "rollout_min_length": 1,
        "rollout_max_length": 1,
        "init_exploration_steps": 100,
    },
    "ContinuousCartPole": {
        "num_epoch": 40,
        "step_per_epoch": 1000,
        "real_step_per_collect": 1,
        "num_train_repeat": 1,
        "dynamics_update_epoch": 1,
        "model_train_freq": 100,
        "model_retain_epochs": 1,
        "rollout_batch_size": int(100e3),
        "rollout_min_length": 1,
        "rollout_max_length": 1,
        "init_exploration_steps": 100,
    },
    "ContinuousPendulum": {
        "num_epoch": 20,
        "step_per_epoch": 1000,
        "real_step_per_collect": 1,
        "num_train_repeat": 1,
        "dynamics_update_epoch": 2,
        "model_train_freq": 100,
        "model_retain_epochs": 1,
        "rollout_batch_size": int(100e3),
        "rollout_min_length": 1,
        "rollout_max_length": 1,
        "init_exploration_steps": 500,
    },
    "ContinuousMountainCar": {
        "num_epoch": 40,
        "step_per_epoch": 1000,
        "real_step_per_collect": 1,
        "num_train_repeat": 1,
        "dynamics_update_epoch": 1,
        "model_train_freq": 100,
        "model_retain_epochs": 1,
        "rollout_batch_size": int(100e3),
        "rollout_min_length": 1,
        "rollout_max_length": 1,
        "init_exploration_steps": 500,
    },
}
