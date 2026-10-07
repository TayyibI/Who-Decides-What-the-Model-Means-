# Who Decides What the Model Means?

Tayyib Ismail

## Introduction

In September 2020, Nature Communications published a paper claiming to track the rise of social trust across Western Europe over five centuries by training a machine learning algorithm to read trustworthiness from the faces in historical paintings (Safra et al). The paper passed peer review at one of the most prestigious general science journals in the world. It reported statistically significant findings, replicated across two independent databases, validated against external metrics. By the formal standards of empirical science, it was technically competent work. By any careful reading, it was also a system that inferred character from facial features, a practice rejected as pseudoscience decades ago, dressed in the language of cognitive science.

The more troubling question is not whether the Safra paper got it wrong, but why this kind of failure keeps happening. Connectionist models held a promise of a strong new methodology in cognitive science. In theory, a gradient descent trained network could find representations of any input domain and generate mappings that learned from experience without the need for explicit rules. The situation is more complicated 30 years later. Some models have given true insights, but many have been withdrawn, corrected, or are red flags. The discovery of failure modes in 1980s developmental modelling has since been rediscovered in recent applications to social behaviour, emotion recognition, hireability, criminality and historical trust. This repetition indicates that these inadequacies are systemic to this modelling style, and not personal mistakes.

This essay notes four common traps: dubious algorithmic decisions being represented as seemingly objective; the secret authorship of findings by the training regime; the naturalisation of created psychological categories via implementation; and the inadequacy of fixes of a purely technical nature. Safra et al. is a prime example of how such pitfalls compound. Based on traditional models of development and contemporary social practice, it is argued in the essay that the basic analytical tools of the module networks only do maths, representations are defined, and most of what the network seems to show is created by modellers. It is necessary to establish these issues before a technical approach to the same.

## The course's approach

The most prevalent theme of the module is that networks only do sums. Numbers are generated, weights are adjusted, and activations are added up. It is the researcher who will interpret such figures and whatever meaning it will have. It is a scientific method and not a device for rhetoric. Any assertion beginning with “the network learns” or “the algorithm shows that” deprives its apparent objectivity, since these terms attribute cognitive functions to a system that lacks them. Goodman gives the philosophical complement in his distinction between resemblance and reference, which is seen in the module via the perspective drawing of Durer: the connection between the output of a model and the phenomenon it is assumed to represent is defined by stipulation, not found by similarity. Neural networks don’t reveal what cognition is like, they produce outputs that we have learned to interpret as corresponding to cognition, often mistaking that constructed correlation for genuine similarity.

It is important since the existing approaches to fairness prevalent in machine learning merely rectify issues that have been established upstream. Birhane and Cummins in their relational ethics paper claim that algorithmic bias cannot be most appropriately explained as a flaw that can be resolved by more powerful loss functions or training data. Algorithms do not merely describe the social environment, but rather create it. They put up classifications, cluster people into those classifications and give them scientific credibility that far outlives the method by which they were created. The architecture remains intact with technical fixes to adjust the model. The content of the course defines the problem at a more advanced level, where the modeller determines the meaning of the numbers, the meaning of the training data, and what conclusions the behaviour of the model can justify.

## Pitfall one: representational laundering.

One such responsible representational design is the NETTalk. The network employed a simple localist encoding of input letters, where there was no natural similarity between letters, which was suitable since English spelling does not provide a straightforward relationship between the shape of letters and their sounds. On the contrary, the output layer employed a well formed distributed representation, which relied on speech dynamics (place, manner and voicing). This was justifiable since phonemes in fact, have real structural connections that domain knowledge could represent. Notably, all design decisions were transparent and subject to consideration.

The common failure mode was, however, revealed in the 1988 New York Times coverage. The network described in the article had neurons and synapses that enabled it to babble like a baby and then slowly learn to say words. The system actually did not map anything but numbers to numbers. The journalists provided all the interpretive overlay in all the psychological language, learning, babbling, pronouncing, etc. This gave a false impression of an artificial baby learning to talk.

Identical falseholds are seen in work that is more technically advanced in modern times. In Safra et al, the authors constructed a sophisticated face processing pipeline with challenged psychological models of trustworthiness and facial action units. They had trained it on artificial white faces, and then used it across centuries of European history portraits to assert that social trust had evolved over the years with GDP and democratisation. Highly debatable assumptions based on a very specific and unrepresentative group of people’s samples and controversial theories of emotions were deeply embedded within the technical system and then lost behind the perceived objectivity of the numbers produced by the model. As Spanton and Guest point out, there is no plausible reason to expect that 16th century viewers shared the trustworthiness beliefs of 21st century undergraduates, nor that synthetic avatars capture the relevant variation in real painted faces. Additionally, a person can look trustworthy and behave deceptively.

This launder pattern is repeated in numerous computer vision studies that try to directly read human social characteristics (criminality, sexual orientation, or trustworthiness) off faces. Scientists program their own hypothetical preferences into the algorithm and make the algorithm generate findings and then frame them as objective facts about the nature of humans, when much of the meaning was in fact programmed by the scientists themselves.

## Pitfall two: the training regime authors the result

A classic warning story is the past tense model of Rumelhart and McClelland. The learning curve of children with English verbs is U shaped: initially, there is correct irregulars followed by overregularisation followed by recovery. This was interpreted as support of two mechanisms (memorisation + rules). The authors trained one network and reported that it was able to reproduce the entire U shape with a single homogeneous learning process. The outcome was a big victory.

The U shape was actually a byproduct of the training regime. The network was initially trained with 10 epochs on the eight irregular and two regular verbs, and then abruptly changed to the entire set of 420 verbs. The overregularisation phase was not the result of the learning of the network but rather due to the discontinuity in the data. This is very deceptive, due to the claim being made and the way it was presented. In addition to that, the original model was never withdrawn and is still quoted.

The main trap is that networks do not generate anything but numbers. It is the modeller who determines what counts as language exposure, what counts as producing a past tense form, and which training epoch represents a given stage of development. These interpretive decisions write the outcome. The network does not find developmental patterns; it gives a base to the hypothesis of the modeller.

This and Safra et al. are based on the same problem. The process of mapping facial action units to trustworthiness scores, extrapolating this to five centuries of portraits, and interpreting the result as historical social trust changes is a sequence of stipulations by the researcher that the model itself does not support.

## Pitfall three: Established categories that are implemented.

Modellers rarely invent the psychological or social constructs they use. Terms like “trustworthiness,” “criminality,” “hireability,” “emotional state,” or “developmental stage” are imported from existing research traditions, each already contested. It is the pitfall of the algorithmic implementation of these constructs into the semblance of natural results. As soon as a network produces a numerical value that is named perceived trustworthiness and the value acts in the same way when applied to different images, the category itself begins to feel objective and stable, as though the model has found it, not made it.

In a word object association task, 14 month old infants did not appear to pay attention to a phonetic alternation between pairs of consonants, unlike 8 month old infants in habituation tests by Stager and Werker. According to this interpretation, the language system undergoes a functional reorganisation between the ages of 8 and 14 months. By analysing this as a basic autoassociator, Schafer and Mareschal showed that continuous learning would produce a similar pattern without any distinct reorganisation. The connection between network and infant depended on a set of conditions. Network error was put into practice as infant looking time, the number of training cycles was defined as developmental time, and the difference between the 8 and 14 month old networks was interpreted as 1,000 and 10,000 training epochs. There is no reference to the months measure in the network. This category, functional reorganisation, was methodologically plausible because there was a model that did not test whether reorganisation actually took place, but only whether it was possible to generate a numerical pattern that resembled the behavioural data using another foundation.

This is the process that Birhane and Cummins refer to as instituting a category by classification, based on Bowker and Star. The fact that a system is called a trustworthiness rater does not simply describe a tool, it makes a practical, citable scientific object. This is demonstrated by Safra et al. Although a 2022 editorial correction in Nature Communications made changes to the title, including the addition of perceived, and noted limitations, the article has been referred to dozens of times, with subsequent research assuming that historical changes in facial trustworthiness is a well established measure instead of a debated construct.

Stark and Hutson take this a step further, stating that computer vision systems used on human faces automatically rekindle physiognomic logic: the belief that internal character or behaviour can be discerned by external physical appearances. These systems replicate the form of long since discredited pseudosciences by reducing a face to a feature vector and mapping it to a social or psychological outcome, whether the researchers intended it or not. Safra et al., and previous studies, such as Wu and Zhang on criminality and Wang and Kosinski on sexual orientation, fall in this tradition. The technical device is secretly reinstating physiognomy with the support of machine learning.

## Pitfall four: the inadequacy of technical fixes

The technical answer to such issues is to diversify the training data, increase the range of raters, fine tune the output labels, tighten statistical constraints, or provide editorial corrections. This is exactly what Nature Communications did to Safra et al. in 2022. They added the word “perceived” to the title and they recognized more limitations and published the reviewer exchange. The revised article is still in the literature. According to Birhane and Cummins, this type of technical fix is structurally unsatisfactory. They note that tweaking methodology is not usually the solution to underlying problems: algorithms are driving the social world they claim to represent, fairness is a moving target, and one should not focus on aggregate performance metrics but on who ends up hurt.

A textbook example of such a limitation is the 2022 correction. It purged the language, but retained the pipeline, the type of historical veracity, and the possibility of damage. The same trend can be observed in the past tense literature: subsequent technical enhancements (e.g., Plunkett and Marchman, 1991) enhanced details, but the essence of the practice, namely letting the training regime write the outcome, was never fundamentally denied. In a word, purely technical solutions address symptoms without changing the underlying modelling approach, and the traps that are common in them. Such problems are not avoided by the technical framework; on the contrary, they are frequently facilitated by it.

## Conclusion

The four pitfalls, representational laundering, training regime authorship, category naturalisation and the ineffectiveness of technical fixes are interrelated and strengthen each other. Collectively, they permit technically qualified, peer reviewable papers, which, however, are structurally unsound. All four can be found in Safra et al., as well as in earlier works such as Rumelhart and McClelland and much of the current work in computer vision of social traits.

We have the necessary means to diagnose these problems at the top: networks do nothing but arithmetic, representations are not found but created, and modellers write much of what the network seems to be telling them. These insights work prior to any technical approach or fairness measures.

The core issue is systemic. The underlying message is that the failures are not owing to the lack of strictness. Even the stringent peer review, such as that in the 2022 correction to Safra et al., could not tackle the upstream issues, whether the categories being measured are sensible, whether the inferential leaps are warranted, and whether such systems should be constructed at all. Mainstream responses usually adapt the terms and introduce restrictions, but leave the fundamental constructs intact.

These pitfalls will continue to be repeated until tools such as those offered by the course become the norm in methodological training, and not an after the fact critique.

## References

Birhane, A., & Cummins, F. (2019). Algorithmic injustices: Towards a relational ethics. arXiv preprint arXiv:1912.07376.

Bowker, G. C., & Star, S. L. (2000). Sorting things out: Classification and its consequences. MIT Press.

Plunkett, K., & Marchman, V. (1991). U shaped learning and frequency effects in a multilayered perceptron: Implications for child language acquisition. Cognition, 38(1), 43–102.

Rumelhart, D. E., & McClelland, J. L. (1986). On learning the past tenses of English verbs. In J. L. McClelland, D. E. Rumelhart, & the PDP Research Group (Eds.), Parallel distributed processing: Explorations in the microstructure of cognition (Vol. 2, pp. 216–271). MIT Press.

Safra, L., Chevallier, C., Grèzes, J., & Baumard, N. (2020). Tracking historical changes in perceived trustworthiness in Western Europe using machine learning analyses of facial cues in paintings. Nature Communications, 11, 4728.

Schafer, G., & Mareschal, D. (2001). Modeling infant speech sound discrimination using simple associative networks. Infancy, 2(1), 7–28.

Sejnowski, T. J., & Rosenberg, C. R. (1987). Parallel networks that learn to pronounce English text. Complex Systems, 1(1), 145–168.

Spanton, R. W., & Guest, O. (2022). Measuring trustworthiness or automating physiognomy? A comment on Safra, Chevallier, Grèzes, and Baumard (2020). arXiv preprint arXiv:2202.08674.

Stager, C. L., & Werker, J. F. (1997). Infants listen for more phonetic detail in speech perception than in word learning tasks. Nature, 388, 381–382.

Stark, L., & Hutson, J. (2022). Physiognomic artificial intelligence. Fordham Intellectual Property, Media & Entertainment Law Journal, 32(4), 922–978.

Wang, Y., & Kosinski, M. (2018). Deep neural networks are more accurate than humans at detecting sexual orientation from facial images. Journal of Personality and Social Psychology, 114(2), 246–257.

Wu, X., & Zhang, X. (2016). Automated inference on criminality using face images. arXiv preprint arXiv:1611.04135.
