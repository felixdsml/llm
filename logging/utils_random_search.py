import random
import dspy
from dspy.evaluate.evaluate_edit import Evaluate
from dspy.teleprompt.teleprompt import Teleprompter
from .bootstrap import BootstrapFewShot
from .vanilla import LabeledFewShot

class BootstrapFewShotWithRandomSearch(Teleprompter):
    def __init__(
        self,
        metric,
        teacher_settings={},
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        num_candidate_programs=16,
        num_threads=6,
        max_errors=10,
        stop_at_score=None,
        metric_threshold=None,
    ):
        self.metric = metric
        self.teacher_settings = teacher_settings
        self.max_rounds = max_rounds
        self.num_threads = num_threads
        self.stop_at_score = stop_at_score
        self.metric_threshold = metric_threshold
        self.min_num_samples = 1
        self.max_num_samples = max_bootstrapped_demos
        self.max_errors = max_errors
        self.num_candidate_sets = num_candidate_programs
        self.max_labeled_demos = max_labeled_demos

        dspy.logger.info(
            "Going to sample between", self.min_num_samples, "and", self.max_num_samples, "traces per predictor.",
        )
        dspy.logger.info("Will attempt to train", self.num_candidate_sets, "candidate sets.")

    def compile(self, student, *, teacher=None, trainset, valset=None, restrict=None, labeled_sample=True):
        self.trainset = trainset
        self.valset = valset or trainset  # TODO: FIXME: Note this choice.

        scores = []
        all_subscores = []
        score_data = []

        for seed in range(-3, self.num_candidate_sets):
            if (restrict is not None) and (seed not in restrict):
                continue

            trainset2 = list(self.trainset)

            if seed == -3:
                # zero-shot
                program2 = student.reset_copy()

            elif seed == -2:
                # labels only
                teleprompter = LabeledFewShot(k=self.max_labeled_demos)
                program2 = teleprompter.compile(student, trainset=trainset2, sample=labeled_sample)

            elif seed == -1:
                # unshuffled few-shot
                program = BootstrapFewShot(
                    metric=self.metric,
                    metric_threshold=self.metric_threshold,
                    max_bootstrapped_demos=self.max_num_samples,
                    max_labeled_demos=self.max_labeled_demos,
                    teacher_settings=self.teacher_settings,
                    max_rounds=self.max_rounds,
                )
                program2 = program.compile(student, teacher=teacher, trainset=trainset2)

            else:
                assert seed >= 0, seed

                random.Random(seed).shuffle(trainset2)
                size = random.Random(seed).randint(self.min_num_samples, self.max_num_samples)

                teleprompter = BootstrapFewShot(
                    metric=self.metric,
                    metric_threshold=self.metric_threshold,
                    max_bootstrapped_demos=size,
                    max_labeled_demos=self.max_labeled_demos,
                    teacher_settings=self.teacher_settings,
                    max_rounds=self.max_rounds,
                )

                program2 = teleprompter.compile(student, teacher=teacher, trainset=trainset2)

            evaluate = Evaluate(
                devset=self.valset,
                metric=self.metric,
                num_threads=self.num_threads,
                max_errors=self.max_errors,
                display_table=False,
                display_progress=True,
            )

            score, subscores = evaluate(program2, return_all_scores=True)
            
            if isinstance(score, dict):
                score = score.get("overall_score", 0)  # or another appropriate key

            all_subscores.append(subscores)

            ############ Assertion-aware Optimization ############
            if hasattr(program2, "_suggest_failures"):
                score = score - program2._suggest_failures * 0.2
            if hasattr(program2, "_assert_failures"):
                score = 0 if program2._assert_failures > 0 else score
            ######################################################

            dspy.logger.info("Score:", score, "for set:", [len(predictor.demos) for predictor in program2.predictors()])

            if len(scores) == 0 or score > max(scores):
                dspy.logger.info("New best score:", score, "for seed", seed)
                best_program = program2

            scores.append(score)
            dspy.logger.info(f"Scores so far: {scores}")

            dspy.logger.info("Best score:", max(scores))

            score_data.append((score, subscores, seed, program2))

            if len(score_data) > 2:  # We check if there are at least 3 scores to consider
                for k in [1, 2, 3, 5, 8, 9999]:
                    top_3_scores = sorted(score_data, key=lambda x: x[0], reverse=True)[:k]

                    # Transpose the subscores to get max per entry and then calculate their average
                    transposed_subscores = list(zip(*[subscores for _, subscores, *_ in top_3_scores if subscores]))                    
                    avg_of_max_per_entry = sum(max(entry) for entry in transposed_subscores) / len(top_3_scores[0][1])

                    dspy.logger.info(f"Average of max per entry across top {k} scores: {avg_of_max_per_entry}")

            if self.stop_at_score is not None and score >= self.stop_at_score:
                dspy.logger.info(f"Stopping early because score {score} is >= stop_at_score {self.stop_at_score}")
                break

        # To best program, attach all program candidates in decreasing average score
        best_program.candidate_programs = score_data
        best_program.candidate_programs = sorted(best_program.candidate_programs, key=lambda x: x[0], reverse=True)

        dspy.logger.info(f"{len(best_program.candidate_programs)} candidate programs found.")

        return best_program
