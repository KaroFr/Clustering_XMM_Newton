# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 10:24:56 CEST 2022

@author: jvperea
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
from astropy.table import Table
from astropy.table import vstack
from multiprocessing import cpu_count

"""
Class for multi-threading processing
"""
class Multithreading:
    
    def __init__(self, threads=1, function=None, input_list=[], **kwargs):
        self.threads    = threads
        self.function   = function
        self.input_list = input_list
        self.kwargs     = kwargs


    """
    Method to run a function in multi threading mode
    Output: result from every thread stacked to a Table (astropy.Table)
    """
    def multi(self):
        threads = self.threads
        fn = self.function
        input_list = self.input_list
        kwargs = self.kwargs

        final_result = Table()
        print(f'\t- Start parallel run ... threads = {threads}')
        with ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_work = {executor.submit(fn, input=input_list[i], **kwargs): i for i
                              in range( len(input_list) )}

            for future in as_completed(future_to_work):
                i = future_to_work[future]
                try:
                    d = future.result()
                    final_result = vstack([final_result, Table(d)])
                except Exception as exc:
                    print(f'\t{i} generated an exception: {exc}')
                else:
                    print(f'\t- Result for thread {i} : ok')
        return final_result


"""
Class for multi-processing
"""
class Multiprocessing:
    
    def __init__(self, threads=1, function=None, input_list=[], **kwargs):
        self.threads    = threads
        self.function   = function
        self.input_list = input_list
        self.kwargs     = kwargs


    """
    Method to run a function in multi processing mode
    Output: result from every thread stacked to a Table (astropy.Table)
    """
    def multi(self):
        threads = min(self.threads, cpu_count())
        fn = self.function
        input_list = self.input_list

        final_result = Table()
        print(f'\t- Start parallel run ... total no of threads = {len(input_list)}')
        print(f'\t- maximum number of threads at the same time = {threads}')
        with ProcessPoolExecutor() as executor:
            results = executor.map(fn, input_list)
            
        for result in results:
            final_result = vstack([final_result, Table(result)])
            
        return final_result


