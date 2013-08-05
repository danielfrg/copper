# -*- coding: utf-8 -*-
from __future__ import division, print_function
import sys
import uuid
from IPython.display import HTML, Javascript, display


class ProgressBar(object):
    ''' IPython Notebook ready text progressbar
    '''
    def __init__(self, max, index=0, desc='', border='|', fill_char='#', width=50):
        self.max = max
        self.index = index
        self.desc = desc
        self.border = border
        self.fill_char = fill_char
        self.width = width
        self.prog_bar = ''
        self.update()

    def next(self):
        if self.index <= self.max:
            self.index += 1
            self.update()

    def animate(self, index):
        self.index = index
        self.update()

    def update(self):
        print('\r', self, end='')
        sys.stdout.flush()

        percent_done = self.index / self.max
        all_full = self.width - 2
        num_hashes = int(round(percent_done * all_full))
        self.prog_bar = self.desc + ' ' + self.border
        self.prog_bar += self.fill_char * num_hashes
        self.prog_bar += ' ' * (all_full - num_hashes)
        self.prog_bar += self.border
        self.prog_bar += '  %.2f%%' % (percent_done * 100)

    def finish(self):
        print('\r', self)

    def __str__(self):
        return str(self.prog_bar)


class JSProgressbar(object):
    ''' IPython notebook ready Javascript ProgressBar
    Uses JQuery UI.

    Note: Depending on the number of iterations and number of bars on the
    same notebooks makes the notebook goes slow.
    Need to work on cleaning the javascript.
    '''
    def __init__(self, max=100, index=0):
        self.divid = str(uuid.uuid4())
        self.max = max
        self.index = index

        pb = HTML( """
            <script src="http://code.jquery.com/ui/1.10.3/jquery-ui.js"></script>
            <div id="{0}" class="progressbar"><div class="progress-label">...</div></div>
            <style>
            .ui-progressbar {{
                position: relative;
                height: 20px;
                width: 70%;
            }}
            .progress-label {{
                position: absolute;
                left: 49.5%;
                top: 2px;
                font-weight: bold;
            }}
            </style>
        """.format(self.divid))
        display(pb)
        self.update()

    def next(self):
        if self.index < self.max:
            self.index += 1
            self.update()

    def animate(self, index):
        self.index = index

    def update(self):
        display(Javascript("""$(function() {{
                            var progressbar = $("#{0}.progressbar");
                            var progressLabel = $("#{0} .progress-label");

                            progressbar.progressbar({{
                                value: {1}
                            }});

                            progressLabel.text({1} + "%" );
                  }});""".format(self.divid, 100 * self.percent())))

    def percent(self):
        return self.index / self.max
