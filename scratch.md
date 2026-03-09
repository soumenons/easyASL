# plan

## ASLLRP ASL Sign Bank
- batch_signs_v1.zip (1.9G)
- batch_signs_v2.zip (1.9G)
- batch_signs_v3.zip (1.9G)
- asllvd_signs_2024_06_27.csv

Sign details CSV:
Columns:
- Video ID number
- **main entry gloss label**	
- **entry/variant gloss label**	
- occurrence label
- start frame of video clip containing the sign (relative to full videos)	
- end frame of video clip containing the sign (relative to full videos)	
- start frame of the sign (relative to full videos)	
- end frame of the sign (relative to full videos)	
- Dominant start handshape	
- Non-dominant start handshape	
- Dominant end handshape	
- Non-dominant end handshape	
- **full video file**	
- **sign type**
- Class Label

Match file to detail information using "full video file" column


Todo:
- [X] Trial with v1 & v2
- [ ] Trial with v1, v2, v3
- [ ] Start with only lexical signs

## dataset
this is the crux of the project, so i want to go through the options i have
- [ ] (WLASL) [Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison](https://dxli94.github.io/WLASL/)
  - Metrics:
    - "the largest video dataset for Word-Level American
Sign Language (ASL) recognition."
    - 2000 different words, 21000+ data points
      - ~~does it contain variations of the same word?~~ [yes it does!](https://huggingface.co/datasets/Voxel51/WLASL/resolve/main/dataset_preview.gif)
  - Issues:
    - [Why Alternative Gloss Labels Will Increase the Value of the WLASL Dataset (2022)](https://www.bu.edu/asllrp/rpt21/asllrp21.pdf)
      - "There is no 1-1 correspondence between sign productions and gloss labels."
        - what this means: gloss labels are the english words associated with physical signs used in computational sign recognition. ASL as it's own language does not have a 1:1 connection with English labels, and the WLASL dataset assumes otherwise.
      - in finding this paper however, i found that they have their own dataset!
      - the paper also goes on to display other elements of the dataset that show the authors lack of understanding the relationship between ASL and English
- [X] [ASLLRP ASL Sign Bank](https://dai.cs.rutgers.edu/dai/s/signbank)
    - Metrics: 
      - 3,542 distinct signs (not including fingerspelled signs, classifiers, or gestures).
- [ ] [MS-ASL: A Large-Scale Data Set and Benchmark for Understanding American Sign Language (2019)](https://arxiv.org/abs/1812.01053)
  - Metrics:
    -  1000 signs in "real-life recording conditions", from 25,000 annotated videos
   - also proposed I3D as a stable architecture for sign-language recognition in the paper
 - [X] [ASL-LEX database](https://asl-lex.org/about.html)
   - Metrics:
     - 2,723 signs
     - lexical database, not video
   - contains lexical and phonological properties, could be interesting to use this information to select a smaller dataset of signs to work with

## other interesting datasets
- [The TUB Sign Language Corpus Collection](https://arxiv.org/html/2508.05374v1)
  -  German Sign Language, Peruvian Sign Language, Costa Rican Sign Language, Colombian Sign Language, Chilean Sign Language, Argentinian Sign Language, Mexican Sign Language
-  [How2Sign Dataset](https://how2sign.github.io/)
   -   First large-scale multimodal and multiview continuous American Sign Language dataset 
   -  good dataset option for attempting continuous sign language recognition?

# ideas

- [ ] ~~ASL practice tool~~
  - GOAL: a beginner tool for simple repetitive practice of ASL (e.g. fingerspelling, numbers) 
  - can incorporate timed repetitions, some kind of progress tracking
  - CONCERNS: though this would be good practice, i don't know if there's a problem that i'm solving here, and the interesting aspect would be the gamification of the tool rather than the training
- [X] reverse ASL dictionary tool
  - GOAL: a tool connected to/trained on a repository of ASL/CSL words, allowing users to sign something that they saw but don't know the meaning of
  - MOTIVATION: 
    - having learned that reverse ASL dictionaries are rare and hard to create, this tool seems more important to work on especially since the issue is real, common, and seems more pressing! this is especially common since there are many words in ASL that have a lot of varieties in signs (e.g. > 6 signs for avocado!)
    - another motivation is that CSL (Canadian Sign Language) resources are harder to come by, and this tool could help in clarifying differences between them
  - CONCERNS: reverse ASL is difficult to implement for many reasons, and i will run into the same issues. one main one being - data! getting a thorough repository of signs will be difficult
    - i could approach this at first with a smaller scope - having reverse search for a small group of signs if available
  
  # thoughts
  - a thought I want to keep in mind throughout any approach is the goal of not creating something that is trying to **teach** ASL. learning ASL from an amazing deaf-first teaching service has really taught me the [importance of learning ASL from people in the deaf community](https://signablevi5ion.com/who-do-we-learn-asl-from/), and no tool would be able to replace or replicate that experience! 
    - instead, i want to use technology to aid in areas that can support the language learning using resources that are created by the deaf community
  - one main takeaway from learning ASL is the importance of looking at ASL in a [lens detached from english](https://www.lifeprint.com/asl101/topics/history8.htm), this is something i want to keep in mind throughout any implementations as well
  - one of the biggest things I want to avoid is becoming yet another ASL fingerspelling project amongst the many online
    - there's nothing wrong with it at all, but it seems like a usage of ASL that is not centering the deaf community, or even the language. and the entire project is often reduced to only fingerspeling as object detection. i would like to do something that goes beyond that, though it may be well beyond my scope!
