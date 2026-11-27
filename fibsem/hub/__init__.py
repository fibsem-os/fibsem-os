"""The hub: a program on a PC away from the microscopes that holds a lab's
list of instruments, reaches each one's gateway over a tunnel it opens, and
shows what each is doing. It only ever connects out: nothing on a microscope
PC calls the hub, and no run depends on it. The door to a microscope is that
PC's gateway (`fibsem.gateway`), which decides what a hub may do there."""
