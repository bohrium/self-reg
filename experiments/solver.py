''' author: samtenka
    change: 2018-08-26
    create: 2018-08-16
    descrp: (approximately) solve a system of rational equations in several variables  
'''

from utils import CC
import numpy as np
import tqdm

expr_types = {'ATOM', 'EXP', 'MUL', 'DIV', 'ADD', 'SUB', 'LOG', 'SQUARE'}

def partial(expression, varname): 
    etype, A, B = expression[0], expression[1], expression[2] if len(expression)==3 else None 
    return {
        'ATOM': (lambda: ['ATOM', 1.0 if A==varname else 0.0]),
        'EXP': (lambda: ['MUL', partial(A, varname), expression]),
        'LOG': (lambda: ['DIV', partial(A, varname), A]),
        'SQUARE': (lambda: ['MUL', partial(A, varname), ['MUL', ['ATOM', 2.0], A]]),
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
        elif self.peek()=='s':
            self.match('square')
            self.match('(')
            tree = ['SQUARE', self.get_expression()]
            self.match(')')
        elif self.peek()=='l':
            self.match('log')
            self.match('(')
            tree = ['LOG', self.get_expression()]
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

def simplify(expression):
    etype, A, B = expression[0], expression[1], expression[2] if len(expression)==3 else None 
    A = simplify(A) if type(A)==type([]) else A
    B = simplify(B) if B is not None else B
    return {
        'ATOM': (lambda: expression),
        'EXP': (lambda:
            ['ATOM', 1.0] if A==['ATOM', 0.0] else
            [etype, A]
        ),
        'LOG': (lambda:
            ['ATOM', 0.0] if A==['ATOM', 1.0] else
            [etype, A]
        ),
        'SQUARE': (lambda:
            ['ATOM', 0.0] if A==['ATOM', 0.0] else
            ['ATOM', 1.0] if A==['ATOM', 1.0] else
            [etype, A]
        ),
        'MUL': (lambda:
            ['ATOM', 0.0] if A==['ATOM', 0.0] else
            ['ATOM', 0.0] if B==['ATOM', 0.0] else
            B if A==['ATOM', 1.0] else
            A if B==['ATOM', 1.0] else
            [etype, A, B]
        ),
        'DIV': (lambda: 
            ['ATOM', 0.0] if A==['ATOM', 0.0] else
            A if B==['ATOM', 1.0] else
            ['ATOM', 1.0] if A==B else
            [etype, A, B]
        ),
        'ADD': (lambda:
            A if B==['ATOM', 0.0] else
            B if A==['ATOM', 0.0] else
            [etype, A, B]
        ),
        'SUB': (lambda:
            A if B==['ATOM', 0.0] else
            ['ATOM', 0.0] if A==B else
            [etype, A, B]
        ),
    }[etype]()

def evaluate(expression, assignments):
    etype, A, B = expression[0], expression[1], expression[2] if len(expression)==3 else None 
    return {
        'ATOM': (lambda: float(assignments[A] if A in assignments else A)),
        'EXP': (lambda: np.exp(evaluate(A, assignments))),
        'LOG': (lambda: np.log(evaluate(A, assignments))),
        'SQUARE': (lambda: evaluate(A, assignments)**2 ),
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
        'LOG': (lambda: 'log({})'.format(string_from(A))),
        'SQUARE': (lambda: '({})**2'.format(string_from(A))),
        'MUL': (lambda: '({}*{})'.format(string_from(A), string_from(B))),
        'DIV': (lambda: '({})/({})'.format(string_from(A), string_from(B))),
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
        return ['SQUARE', exprs[0]]
    else:
        return ['ADD', sum_squares(exprs[0:1]), sum_squares(exprs[1:])]

def solve(constraints, start_eta=0.1, mom_decay=0.95, start_noise=1.0, random_tries=10000, grad_tries=1000): 
    expr = simplify(  sum_squares([['SUB', left, right] for (left, right) in constraints])  )
    fvs = free_vars(expr)
    partials = {v:partial(expr, v) for v in fvs}

    best_assignments = {v: start_noise*np.random.randn() for v in fvs} 
    best_val = evaluate(expr, best_assignments) 

    assignments = best_assignments
    noises = {v: start_noise for v in fvs}
    for i in range(random_tries):
        assignments = {v: noises[v] * np.random.laplace() for v in fvs}
        val = evaluate(expr, assignments) 
        noises = {v: noises[v]*0.99 + start_noise*0.01 for v in fvs}
        if val <= best_val:
            best_val, best_assignments = val, assignments
            noises = {v: noises[v]*0.2 + abs(best_assignments[v])*0.8 for v in fvs}

    eta = start_eta
    moms = {v: 0.0 for v in fvs}
    for i in tqdm.tqdm((range(grad_tries))):
        partial_vals = {v: evaluate(partials[v], best_assignments) for v in fvs}
        moms = {v: moms[v]*mom_decay + partial_vals[v]*(1.0-mom_decay) for v in fvs}
        assignments = {v: best_assignments[v] - eta * moms[v] for v in fvs} 
        val = evaluate(expr, assignments) 
        if val <= best_val:
            best_val, best_assignments = val, assignments
            eta *= 4.0/3
        else:
            eta *= 2.0/3

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
            print(CC + '    '+'  '.join('@M {}@C  = @G {:.7f}@C '.format(v, a[v]) for v in a))
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
    constraints = [(Parser(left).get_tree(), Parser(right).get_tree()) for left,right in handsides]
    assignments, val = solve(constraints) 
    return assignments, val

if __name__=='__main__':
    interactive()

    a, val = solve_from_strings(['A*A=A+1', 'A=B*B'])
    print(a)
    print(val)

