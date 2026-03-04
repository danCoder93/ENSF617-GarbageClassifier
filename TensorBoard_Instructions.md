
# TensorBoard Instructions

1. On the remote server, start TensorBoard bound to localhost (important):

```sh
tensorboard --logdir /path/to/runs --host 127.0.0.1 --port 6006
```

1. On your laptop (local machine), open an SSH tunnel:

```sh
ssh -L 6006:127.0.0.1:6006 your_user@remote_host
```

1. Open in your local browser:

```sh
http://127.0.0.1:6006
```
