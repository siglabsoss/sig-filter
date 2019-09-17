
One time run this:
===
Run this one time, and then run the next section
```
sudo apt install python3-dev
sudo -H pip3 install -U virtualenv
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```


Every other time run this:
===
Only run this section if you're all setup
```
source ./venv/bin/activate
jupyter lab
```




Installing a new pip
===
If you need more pip packages, do this.  Make sure it says `(venv)` on your terminal, if not run the previous section


```
pip install somepackage
pip -l freeze > requirements.txt
```

And the commit the requirements.txt








See Also
===


https://pip.pypa.io/en/stable/installing/
