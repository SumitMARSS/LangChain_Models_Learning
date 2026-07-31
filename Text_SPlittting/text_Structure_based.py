from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

text = """
Cricket is a bat-and-ball game played between two teams of eleven players on a cricket field, at the centre of which is a rectangular 22-yard-long pitch with a wicket, a set of three wooden stumps sited at each end. One team, designated the batting team, attempts to score as many runs as possible, whilst their opponents field. Each phase of play is called an innings. After either ten 
batsmen have been dismissed or a set number of overs have been completed, the innings ends and the two
 teams then swap roles. The winning team is the one that scores the most runs, including any extras
  gained, during their period batting. At the start of each game, two batsmen and eleven fielders enter the field of play. 
  The play begins when a designated member of the fielding team, known as the bowler, delivers the ball from one end of the pitch to the other, 
  towards a set of wooden stumps, in front of which stands one of the batsmen, known as the striker. The striker's role is to prevent the ball from hitting the stumps through use of his bat, 
  and simultaneously strike it sufficiently well to score runs. The other batsman, known as the non-striker, waits at the opposite end of the pitch by the bowler. 
  The bowler's intention is to both prevent the scoring of runs and to dismiss the batsman, at which point the dismissed batsman has to leave the field and another teammate replaces him at the crease. 
  The most common forms of dismissal are bowled, when the bowler hits the stumps directly with the ball, leg before wicket, when the batsman prevents the ball from hitting the stumps with his body instead 
  of his bat, and caught, when the batsman hits the ball into the air and it is intercepted by a fielder before touching the ground. Runs are scored through two main methods: either hitting the ball sufficiently powerfully that it crosses the boundary, or through the two batsmen swapping ends by each simultaneously running the length of the pitch in opposite directions whilst the fielders are retrieving the ball. If a fielder is able to retrieve the ball sufficiently quickly and put down the wicket with either batsman out of his ground, a run-out occurs. 
"""

spplitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)

chunks = spplitter.split_text(text)
print(len(chunks))
print(chunks[0])
print()
print(chunks)