# Challenge Ideas for the Hackathon Session

## Run on Bridges-2

If you were able to obtain an account on Bridges-2, here are some things you can do there to experience the multi-node
behaviors of Dragon.

When you log on, here are the steps to get your environment set correctly:

```bash
. $PROJECT/../shared/bin/setup
```

From there, you can submit interactive jobs using the `interact`. For example:

```bash
interact -p RM -N 2
```

Don't forget to source the `setup` file as above again once you get an interactive session.

### Examine the resources Dragon see across nodes

Run the `resources.py` script to information about the nodes Dragon is on. Try looking at more of the available
[`Node`](https://dragonhpc.github.io/dragon/doc/_build/html/ref/native/dragon.native.machine.html#dragon.native.machine.Node) properties in the documention.

```bash
dragon resources.py
```

### Run `multiprocessing` across nodes

Look at the examples in the Dragon
[multiprocessing examples](https://dragonhpc.github.io/dragon/doc/_build/html/cbook/multiprocessing.html) and
try running them.

# Create new examples

We love new example code! Create your own and create a pull request on the (Dragon repo)[https://github.com/DragonHPC/dragon].