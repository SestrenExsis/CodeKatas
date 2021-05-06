class Solver:
    def solve(self, ticket_count, tickets_sold):
        # Create intervals to represent runs of unsold tickets
        unsold_tickets = []
        prev_ticket = 0
        for ticket in sorted(tickets_sold):
            if ticket > prev_ticket + 1:
                unsold_tickets.append([prev_ticket + 1, ticket - 1])
            prev_ticket = ticket
        if prev_ticket < ticket_count:
            unsold_tickets.append([prev_ticket + 1, ticket_count])
        # Generate set of tickets adjacent to sold tickets (best candidates)
        candidates = set()
        for ticket in tickets_sold:
            if (
                ticket - 1 not in tickets_sold and
                ticket - 1 >= 1
            ):
                candidates.add(ticket - 1)
            if (
                ticket + 1 not in tickets_sold and
                ticket + 1 <= ticket_count
            ):
                candidates.add(ticket + 1)
        # Try every combination of two candidate cards
        max_hits = 0
        for a in candidates:
            for b in candidates:
                hits = 0
                for start, end in unsold_tickets:
                    if any([
                        start == 1 and end in (a, b),
                        start in (a, b) and end == ticket_count,
                        start in (a, b) and end in (a, b),
                    ]):
                        hits += end - start + 1
                    elif start in (a, b) or end in (a, b):
                        hits += 1 + (end - start) // 2
                max_hits = max(max_hits, hits)
        result = max_hits / ticket_count
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            _, ticket_count = tuple(map(int, input().split(' ')))
            tickets_sold = set(map(int, input().split(' ')))
            solution = self.solve(ticket_count, tickets_sold)
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