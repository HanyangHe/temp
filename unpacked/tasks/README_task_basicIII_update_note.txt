Basic-III task file update note
===============================

Do not overwrite task_basicIII.m if it contains your custom Basic-III target
formula. The recommended new-dictionary architecture can be applied in the demo
after loading the task:

  task = task_basicIII();
  task = configure_newdict_arch(task, 'basicIII', 'baseline');

For an interaction-dictionary stress test:

  task = task_basicIII();
  task = configure_newdict_arch(task, 'basicIII', 'interaction');

The only required architecture changes are:
- remove 'id' from task.arch.opNames;
- set task.arch.interact.enable = false for the conservative baseline;
- optionally enable interaction terms for an enhanced test.
