class Solver:
    def get_hits(self, unsold_tickets, ticket_count, a, b):
        hits = 0
        for start, end in unsold_tickets:
            if any((
                start == 1 and end in (a, b),
                end == ticket_count and start in (a, b),
                start in (a, b) and end in (a, b),
            )):
                hits += end - start + 1
            elif start in (a, b) or end in (a, b):
                hits += 1 + (end - start) // 2
        result = hits
        return result
    
    def solve(self, ticket_count, tickets_purchased):
        # Generate intervals to represent unsold tickets
        unsold_tickets = []
        prev_ticket = 0
        tickets = list(sorted(tickets_purchased))
        for ticket in tickets:
            if ticket != prev_ticket + 1:
                unsold_tickets.append([prev_ticket + 1, ticket - 1])
            prev_ticket = ticket
        if prev_ticket < ticket_count:
            unsold_tickets.append([prev_ticket + 1, ticket_count])
        # The best tickets to buy are adjacent to purchased tickets
        candidates = set()
        for ticket in tickets_purchased:
            if (
                ticket > 1 and
                ticket - 1 not in tickets_purchased
            ):
                candidates.add(ticket - 1)
            if (
                ticket < ticket_count and
                ticket + 1 not in tickets_purchased
            ):
                candidates.add(ticket + 1)
        # Try all combinations from the candidate tickets
        max_odds = 0
        for a in candidates:
            for b in candidates:
                hits = self.get_hits(unsold_tickets, ticket_count, a, b)
                odds = hits / ticket_count
                max_odds = max(max_odds, odds)
        result = max_odds
        return result
    
    def main(self):
        '''
        For Test Set 1:
            1 <= len(tickets_purchased) <= 30
            1 <= ticket_count <= 30
        For Test Set 2:
            1 <= len(tickets_purchased) <= 30
            1 <= ticket_count <= 10 ** 9
        '''
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            N, ticket_count = tuple(map(int, input().split(' ')))
            tickets_purchased = set(tuple(map(int, input().split(' '))))
            solution = self.solve(ticket_count, tickets_purchased)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

if __name__ == '__main__':
    '''
    Usage
    python SolverSandbox.py < inputs/SolverA.in
    '''
    solver = Solver()
    solver.main()