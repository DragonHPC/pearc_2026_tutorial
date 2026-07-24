# Troubleshooting Guide

If you happen to make a mistake and get into a bad state within the Jupyter notebook you
can follow this guide for some helpful hints to getting things working again.

# Tip

The tutorials are in Jupyter notebook form. If you make a mistake while writing
code you may leave your notebook in a bad state. If that happens you can restart
the kernel to recover from it. Go to `Kernel` and select `Restart Kernel...`.

After restarting the kernel, you can re-open the notebook and start over. Just
make sure to re-execute any startup code that was necessary for the tutorial.
Execute any cells that are a prerequisite for your current exercise, then you
should be good to go!

# Still not working?

If using VS Code you can shutdown the Jupyter Server and restart VS Code. That should restart the Jupyter Server for you.

If you are still having issues you can clean up by typing `dragon-cleanup` in the
terminal window and then try restarting by running `dragon-jupyter`.

If that does not work you can then examine processes
with `ps aux | grep -E 'jupyter'` and you can kill them all with
`pkill -9 -f jupyter`.

Then run `dragon-jupyter` again.

After running `dragon-jupyter` look for the line that looks like this in the terminal window.

```bash
http://127.0.0.1:8888/tree?token=e335a788b94549b8113efa59c098948e911221e56e34cdbf
```

Paste your similar line it into your browser. If it still asks for a token then
copy your token from after the `?token=` and paste that into the **Log in**
field.

