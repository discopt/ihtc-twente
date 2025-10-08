# ihtc-twente

This is the code for the [Integrated Healthcare Timetabling Competition 2024](https://ihtc2024.github.io/) participation of **Team Twente**.
Our team consists of

- [Daniela Guericke](https://people.utwente.nl/d.guericke)
- [Rolf van der Hulst](https://people.utwente.nl/r.p.vanderhulst)
- [Asal Karimpour](https://people.utwente.nl/a.karimpour)
- [Ieke Schrader](https://people.utwente.nl/i.m.w.schrader)
- [Matthias Walter](https://people.utwente.nl/m.walter)

## Approach ##

Our algorithm is a combination of mixed-integer programming (via Gurobi), constraint programming (via OR-Tools) and simulated annealing.
Details can soon be found on arXiv.

## Usage ##

You can run `main.py [OPTIONS] INSTANCE-FILE [OUTPUT-FILE]`
with the following options:

- `-Ttotal <NUM>` Number of seconds for overall algorithm.
- `-Tfinal <NUM>` Number of seconds for final phase.
- `-Nroom <NUM>`  Number of failed room attempts before reducing capacity; default: 6.
- `-hlocal`       Heuristic shall be local search.
- `-hinit`        Heuristic shall be initialized from current solution if it exists.

