import heapq

class Solver:
    '''
    100 players
    10_000 questions
    question difficulties and player skills are random with range [-3.00, 3.00]
    f(x) = 1 / (1 + e ^ -x), where x = skill - difficulty

    The cheater is the person whose odds of getting a hard question right 
    is least influenced by increases in a question's difficulty
    '''
    player_count = 100
    question_count = 10_000
    hard_question_threshold = 2_500

    def solve(self, player_answers):
        difficulties = [0] * self.question_count
        for player_id, answers in enumerate(player_answers):
            for question_id, answer in enumerate(answers):
                if answer == '0':
                    difficulties[question_id] += 1
        questions = []
        for question_id, difficulty in enumerate(difficulties):
            heapq.heappush(questions, (-difficulty, question_id))
        hard_questions = []
        for _ in range(self.hard_question_threshold):
            _, question_id = heapq.heappop(questions)
            hard_questions.append(question_id)
        while len(questions) > self.hard_question_threshold:
            _, question_id = heapq.heappop(questions)
        easy_questions = []
        while len(questions) > 0:
            _, question_id = heapq.heappop(questions)
            easy_questions.append(question_id)
        quiz = []
        for player_id in range(self.player_count):
            easy_questions_correct = 0
            for question_id in easy_questions:
                if player_answers[player_id][question_id] == '1':
                    easy_questions_correct += 1
            hard_questions_correct = 0
            for question_id in hard_questions:
                if player_answers[player_id][question_id] == '1':
                    hard_questions_correct += 1
            quiz.append((easy_questions_correct, hard_questions_correct))
        cheater = None
        lowest_ratio = float('inf')
        for player_id in range(self.player_count):
            easy_questions_correct, hard_questions_correct = quiz[player_id]
            ratio = easy_questions_correct / hard_questions_correct
            if ratio < lowest_ratio:
                cheater = player_id
                lowest_ratio = ratio
        result = cheater + 1
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            threshold = int(input())
            player_answers = []
            for i in range(self.player_count):
                player_answers.append(input())
            solution = self.solve(player_answers)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

if __name__ == '__main__':
    solver = Solver()
    solver.main()