import sys
import dedalus.tools.post as post

filepath = sys.argv[-1]
post.merge_process_files_single_set(filepath)
