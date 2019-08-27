''' author: samtenka
    change: 2018-08-26
    create: 2018-08-16
    descrp: (approximately) solve a system of rational equations in several variables  
'''

from utils import CC
import numpy as np

expr_types = {'ATOM', 'EXP', 'MUL', 'DIV', 'ADD', 'SUB'}

def partial(expression, varname): 
    etype, A, B = expression[0], expression[1], expression[2] if len(expression)==3 else None 
    return {
        'ATOM': (lambda: ['ATOM', 1.0 if A==varname else 0.0]),
        'EXP': (lambda: ['MUL', partial(A, varname), expression]),
        'MUL': (lambda: [
            'ADD',
                ['MUL', partial(A, varname), B],
                ['MUL', A, partial(B, varname)]
        ]),
        'DIV': (lambda: [
            'ADD',
                ['DIV', partial(A, varname), B],
                ['MUL', ['ATOM', -1.0], ['DIV', ['MUL', A, partial(B, varname)], ['MUL', B, B]]]
        ]),
        'ADD': (lambda: ['ADD', partial(A, varname), partial(B, varname)]),
        'SUB': (lambda: ['SUB', partial(A, varname), partial(B, varname)])
    }[etype]()

class Parser:
    def __init__(self, string):
        self.string = string.replace(' ', '')
        self.i=0
    def get_tree(self):
        tree = self.get_expression()
        assert self.i==len(self.string)
        return tree

    def peek(self):
        return self.string[self.i] if self.i!=len(self.string) else '\0'
    def match(self, s):
        assert self.string[self.i:self.i+len(s)]==s
        self.i+=len(s)
    def march(self):
        self.i+=1

    def get_number(self):
        old_i = self.i
        if self.peek() in '+-': self.march()
        while self.peek() in '0123456789.': self.march()
        return ['ATOM', float(self.string[old_i:self.i])]
    def get_identifier(self): 
        old_i = self.i
        while self.peek() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ': self.march()
        return ['ATOM', self.string[old_i:self.i]]
    def get_factor(self):
        if self.peek()=='(':
            self.match('(')
            tree = self.get_expression()
            self.match(')')
        elif self.peek()=='-':
            self.match('-')
            factor = self.get_factor()
            tree = ['SUB', ['ATOM', 0.0], factor]
        elif self.peek()=='d':
            self.match('d_')
            direction = self.get_identifier()[1] 
            self.match('(')
            expression = self.get_expression()
            self.match(')')
            tree = partial(expression, direction)
        elif self.peek()=='e':
            self.match('exp')
            self.match('(')
            tree = ['EXP', self.get_expression()]
            self.match(')')
        elif self.peek() in '+-0123456789.':
            tree = self.get_number()
        else:
            tree = self.get_identifier()
        return tree
    def get_term(self):
        factorA = self.get_factor() 
        if self.peek()=='*':
            self.match('*')
            tree = ['MUL', factorA, self.get_term()]
        elif self.peek()=='/':
            self.match('/')
            tree = ['DIV', factorA, self.get_term()]
        else:
            tree = factorA
        return tree
    def get_expression(self):
        termA = self.get_term() 
        if self.peek()=='+':
            self.match('+')
            tree = ['ADD', termA, self.get_expression()]
        elif self.peek()=='-':
            self.match('-')
            tree = ['SUB', termA, self.get_expression()]
        else:
            tree = termA
        return tree

def evaluate(expression, assignments):
    etype, A, B = expression[0], expression[1], expression[2] if len(expression)==3 else None 
    return {
        'ATOM': (lambda: float(assignments[A] if A in assignments else A)),
        'EXP': (lambda: np.exp(evaluate(A, assignments))),
        'MUL': (lambda: evaluate(A, assignments) * evaluate(B, assignments)),
        'DIV': (lambda: evaluate(A, assignments) / evaluate(B, assignments)),
        'ADD': (lambda: evaluate(A, assignments) + evaluate(B, assignments)),
        'SUB': (lambda: evaluate(A, assignments) - evaluate(B, assignments)),
    }[etype]()

def string_from(expression):
    etype, A, B = expression[0], expression[1], expression[2] if len(expression)==3 else None 
    return {
        'ATOM': (lambda: str(A)),
        'EXP': (lambda: 'exp({})'.format(string_from(A))),
        'MUL': (lambda: '{}*{}'.format(string_from(A), string_from(B))),
        'DIV': (lambda: '{}/({})'.format(string_from(A), string_from(B))),
        'ADD': (lambda: '{} + {}'.format(string_from(A), string_from(B))),
        'SUB': (lambda: '{} - ({})'.format(string_from(A), string_from(B))),
    }[etype]()

def free_vars(expression):
    etype, A, B = expression[0], expression[1], expression[2] if len(expression)==3 else None 
    if etype=='ATOM':
        return {A} if type(A)==type('') else set([])
    elif B is None:
        return free_vars(A)
    else:
        return free_vars(A).union(free_vars(B))

def sum_squares(exprs):
    if len(exprs)==0:
        return ['ATOM', 0.0]
    elif len(exprs)==1:
        return ['MUL', exprs[0], exprs[0]] 
    else:
        return ['ADD', sum_squares(exprs[0:1]), sum_squares(exprs[1:])]

def solve(constraints, tolerance = 1e-12, eta = 0.01, init_noise=0.0, drift_noise=1.0, tries=100, steps_per_try=1000): 
    expr = sum_squares([['SUB', left, right] for (left, right) in constraints])
    fvs = free_vars(expr)
    partials = {v:partial(expr, v) for v in fvs}

    best_val = float('+inf')
    best_assignments = None
    
    for i in range(tries):
        assignments = {v: init_noise*np.random.randn() for v in fvs}
        for j in range(steps_per_try):
            val = evaluate(expr, assignments) 

            new_assignments = {v: assignments[v] - eta * evaluate(partials[v], assignments) for v in fvs} 
            newval = evaluate(expr, new_assignments) 
            if newval <= val:
                val, assignments = newval, new_assignments

            new_assignments = {v: assignments[v] - drift_noise*np.random.randn() for v in fvs} 
            newval = evaluate(expr, new_assignments) 
            if newval <= val:
                val, assignments = newval, new_assignments
            
        if val <= best_val:
            best_val, best_assignments = val, assignments

        if val<tolerance:
            break

    return best_assignments, best_val

def interactive(): 
    print()
    print(CC + 'hello! i can help you solve systems of equations!')
    print(CC + 'please enter some equations (with uppercase variable names)')
    print(CC + 'for example, try entering @M A*A = A+1@C  and @M A=B*B@C ')
    print(CC + 'enter `@Y exit@C ` to exit, `@Y solve@C ` to solve, `@Y print@C ` to show the system so far, or `@Y clear@C ` to clear the system')
    constraints = [] 
    while True:
        command = input('>> ')
        if command=='':
            pass
        elif command=='exit':
            exit()
        elif command=='solve':
            a, val = solve(constraints)
            print(CC + '    '+'  '.join('@M {}@C =@G {:.7f}@C '.format(v, a[v]) for v in a))
            print(CC + '    accurate to @R {:.7f}@C '.format(val**0.5))
        elif command=='clear':
            constraints = []
        elif command=='print':
            for left, right in constraints:
                print(CC + '@M {}@C =@M {}@C '.format(string_from(left), string_from(right)))
        else:
            try:
                left, right = command.split('=') 
                constraints.append((Parser(left).get_tree(), Parser(right).get_tree()))
            except ValueError:
                print(CC + '@R uh oh! I did not understand that equation@C ')

def solve_from_strings(system):
    ''' `system` is a list of strings '''
    handsides = [equation.split('=') for equation in system] 
    constraints = [((Parser(left).get_tree(), Parser(right).get_tree())) for left,right in handsides]
    assignments, val = solve(constraints) 
    return assignments, val

if __name__=='__main__':
    interactive()

    a, val = solve_from_strings(['A*A=A+1', 'A=B*B'])
    print(a)
    print(val)

