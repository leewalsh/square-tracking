{-# LANGUAGE TupleSections #-}

import Data.Function (on)
import Data.List
import Data.Maybe

type Time = Int
type Dist = Double
type TrackID = Int
type Point = (Double,Double)
type DataPoints = [(Time,Point)]
type TrackPt = ((Time, Point), TrackID)
type Track = [(Time, Point)]

head' :: [a] -> Maybe a
head' (x:_) = Just x
head' _     = Nothing

findTracks :: Time -> Dist -> [(Time, Point)] -> Track
findTracks maxT maxD pts = go pts []
    where go :: [(Time, Point)] -> [Track] -> [Track]
          go (pt:pts) trks = map head' trks
                    
              let options =
                  filter (\(_,dist) -> dist < maxD)
                  $ sortBy (compare `on` snd)
                  $ map (\trk -> (addPointToTrack pt trk, pointDistFromTrack pt trk) ) trks
              in case options of
                  [] -> go pts (tail pts)

distance :: Point -> Point -> Dist
distance (x,y) (x',y') = sqrt $ (x-x')^2 + (y-y')^2



addPointToTrack :: Point -> Tack -> Track
pointDistFromTrack :: Point -> Track -> Dist

buildTracks :: Dist -> TrackID -> [TrackPt] -> DataPoints -> [TrackPt]
buildTracks maxDist lastId trks [] = trks
buildTracks maxDist lastId trks ((t,pt):pts) =
    let comparePoints :: TrackPt -> TrackPt -> Ordering
        comparePoints ((t1,p1),_) ((t2,p2),_) =
            (t - t1, p1 `distance` pt) `compare` (t - t2, p2 `distance` pt)

        trackPts :: [TrackPt]
        trackPts = filter (\((t,p),_)->p `distance` pt < maxDist)
              $ sortBy comparePoints trks
       in case trackPts of
           ((_,_),tid):_  -> buildTracks maxDist lastId (((t,pt), tid):trks) pts
           []             -> buildTracks maxDist (lastId+1) (((t,pt), lastId):trks) pts

points = concat [ map (1,) [(1,1), (10,20), (50,50)]
                , map (2,) [(2,2), (11,21), (100,100)]
                , map (3,) [(100,99)] ] 
main = do
    mapM_ print $ groupBy ((==) `on` snd) $ sortBy (compare `on` snd) $ buildTracks 10 0 [] points

