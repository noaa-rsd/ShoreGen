import os
import tarfile
import filesizeformat
import datetime


DATA_dir = r'Z:\\'
os.chdir(DATA_dir)

tarFiles = [tf for tf in os.listdir(DATA_dir) \
            if tf.endswith("tar.gz")]

# for each segment
for tf in tarFiles:

    untar_file = os.path.join(DATA_dir, tf)
    untar_path = os.path.join(DATA_dir, tf.replace(".tar.gz", ""))
    
    tarFile = tarfile.open(untar_file)
    tarMembers = tarFile.getnames()
    print("Start time: ")
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print("Extracting " + str(len(tarMembers)) + " files from " + untar_file)
    print("\tto " + untar_path)

    os.makedirs(untar_path)
    os.chdir(untar_path)
    untar_count = 0
    for band in tarFile.getmembers():
        untar_count += 1
        if not(os.path.exists(band.name)):
            print("\t(" + str(untar_count) + "/" + str(len(tarMembers)) + \
                  ") Extracting " + band.name + "...",
            tarFile.extractall(members=[tarFile.getmember(band.name)]))
            print("done")
        else:
            extractedSize = os.stat(band.name).st_size
            if band.size != extractedSize:
                print("\t(" + str(untar_count) + "/" + str(len(tarMembers)) + \
                      ") Re-extracting " + band.name + " (only " + \
                      str(filesizeformat.filesizeformat(extractedSize)) + \
                      " of " + str(filesizeformat.filesizeformat(band.size)) + \
                      " previously extracted)...",
                tarFile.extractall(members=[tarFile.getmember(band.name)]))
                print("done")
            else:
                print("\t(" + str(untar_count) + "/" + str(len(tarMembers)) + \
                      ") " + band.name + " already exists")

    tarFile.close()

    # delete tar when it's untar'd
    # print("Successfully untarred " + untar_file + ", deleting it...",
    # os.remove(untar_file)
    print("done")
    print("End time: ")
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S') + "\n")