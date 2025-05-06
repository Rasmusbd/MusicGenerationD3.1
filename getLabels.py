import csv

"""Functions to get genre/label of a track.
   Use it by importing these functions below"""

def createGenreMap():
    genreMap = {} #key:track id, value:genre

    # Open the CSV file in read mode
    with open('labels.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        #Skip first row (header row)
        next(csv_reader)
    
    # Iterate through the rows in the CSV file
    for row in csv_reader:
        raw_number = int(row[0])
        track_id = f"{raw_number:06d}"
        genre = row[1]

        genreMap[track_id] = genre

    return genreMap
    
def getLabel(genreMap, track_id):
    #Genre map is dictionary created above, track_id is string id of track.
    return genreMap[track_id]