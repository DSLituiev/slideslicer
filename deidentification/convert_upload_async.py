
# coding: utf-8
# parts of code are borrowed from:
# https://fredrikaverpil.github.io/2017/06/20/async-and-await-with-subprocesses/

import pandas as pd
import os
from os import makedirs
from os.path import dirname, join as pj
from shutil import copyfile
from subprocess import PIPE, call
import sys
sys.path.append('../anonymize_slide/')
from anonymize_slide import do_aperio_svs
import openslide

import time
import platform
import asyncio



async def anonymize_put_on_gcloud(oldpath, newpath):
    print('='*30)
    outdir = pj(outparentdir, dirname(newpath))
    outfile = pj(outparentdir, newpath)
    routfile = '/'.join(outfile.split('/')[1:])

    if not os.path.exists(outdir):
        makedirs(outdir)
    copyfile(pj(inparentdir, oldpath), outfile)
    # remove the label
    do_aperio_svs(outfile)
    # make sure no label
    slide = openslide.OpenSlide(outfile)
    assert 'label' not in list(slide.associated_images)

    await run_command_shell(" ".join(["gsutil","cp", "-r", outfile, "gs://kidney-rejection/" + routfile]))
    await run_command_shell((["rm", "-r", outfile]))



async def run_command(*args):
    """Run command in subprocess
    
    Example from:
        http://asyncio.readthedocs.io/en/latest/subprocess.html
    """
    # Create subprocess
    process = await asyncio.create_subprocess_exec(
        *args,
        # stdout must a pipe to be accessible as process.stdout
        stdout=asyncio.subprocess.PIPE)

    # Status
    print('Started:', args, '(pid = ' + str(process.pid) + ')')

    # Wait for the subprocess to finish
    stdout, stderr = await process.communicate()

    # Progress
    if process.returncode == 0:
        print('Done:', args, '(pid = ' + str(process.pid) + ')')
    else:
        print('Failed:', args, '(pid = ' + str(process.pid) + ')')

    # Result
    result = stdout.decode().strip()

    # Return stdout
    return result


async def run_command_shell(command):
    """Run command in subprocess (shell)
    
    Note:
        This can be used if you wish to execute e.g. "copy"
        on Windows, which can only be executed in the shell.
    """
    if isinstance(command, list):
        command = " ".join(command)
    # Create subprocess
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE)

    # Status
    print('Started:', command, '(pid = ' + str(process.pid) + ')')

    # Wait for the subprocess to finish
    stdout, stderr = await process.communicate()

    # Progress
    if process.returncode == 0:
        print('Done:', command, '(pid = ' + str(process.pid) + ')')
    else:
        print('Failed:', command, '(pid = ' + str(process.pid) + ')')

    # Result
    result = stdout.decode().strip()

    # Return stdout
    return result


def make_chunks(l, n):
    """Yield successive n-sized chunks from l.

    Note:
        Taken from https://stackoverflow.com/a/312464
    """
    if sys.version_info.major == 2:
        for i in xrange(0, len(l), n):
            yield l[i:i + n]
    else:
        # Assume Python 3
        for i in range(0, len(l), n):
            yield l[i:i + n]


def run_asyncio_commands(tasks, max_concurrent_tasks=0):
    """Run tasks asynchronously using asyncio and return results

    If max_concurrent_tasks are set to 0, no limit is applied.

    Note:
        By default, Windows uses SelectorEventLoop, which does not support
        subprocesses. Therefore ProactorEventLoop is used on Windows.
        https://docs.python.org/3/library/asyncio-eventloops.html#windows
    """

    all_results = []

    if max_concurrent_tasks == 0:
        chunks = [tasks]
    else:
        chunks = make_chunks(l=tasks, n=max_concurrent_tasks)

    for tasks_in_chunk in chunks:
        if platform.system() == 'Windows':
            loop = asyncio.ProactorEventLoop()
            asyncio.set_event_loop(loop)
        else:
            loop = asyncio.get_event_loop()

        commands = asyncio.gather(*tasks_in_chunk)  # Unpack list using *
        results = loop.run_until_complete(commands)
        all_results += results
#         loop.close()
    return all_results


if __name__ == '__main__':
    import yaml
    with open("upload-config.yaml") as fh:
        config = yaml.load(fh)
        config
    
    inparentdir = config["inparentdir"]
    outparentdir = config["outparentdir"] 
    fnmap = config["fnmap"]
    max_concurrent_tasks = config["max_concurrent_tasks"]

    dfmap = pd.read_table(fnmap)
    tasks = []
    for nn,(kk, ds) in enumerate(dfmap.iterrows()):
        tasks.append(anonymize_put_on_gcloud(ds["filepath"], ds["newpath"]))


    run_asyncio_commands(tasks, max_concurrent_tasks=max_concurrent_tasks)

    print("removing the temporary directory")
    run_asyncio_commands([ run_command_shell((["rm", "-r", outparentdir])) ])


