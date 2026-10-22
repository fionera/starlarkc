/*
 * Starlark LLVM runtime
 *
 * This is deliberately a small, leak-friendly runtime: values are allocated
 * with malloc and freed by process exit. That keeps the generated ABI simple
 * while still giving compiled Starlark programs normal native execution.
 */
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

typedef struct SkValue {
    int64_t tag;
    int64_t payload;
} SkValue;

enum {
    SK_NONE = 0,
    SK_BOOL = 1,
    SK_INT = 2,
    SK_FLOAT = 3,
    SK_STRING = 4,
    SK_LIST = 5,
    SK_TUPLE = 6,
    SK_DICT = 7,
    SK_RANGE = 8,
    SK_FUNCTION = 9,
    SK_BUILTIN = 10,
    SK_UNBOUND = 11,
    SK_METHOD = 12,
    SK_STRUCT = 13,
    SK_STRING_ELEMS = 14,
    SK_CHAR = 15,
    SK_INT_LIST = 16
};

typedef struct SkString {
    int64_t len;
    char data[];
} SkString;

typedef struct SkList {
    int64_t len;
    int64_t cap;
    SkValue *items;
} SkList;

/* A compact list of unboxed integers.  It has the same three-word header as
 * SkList so the compiler can use one GEP layout for both, but `items` points
 * at an int64_t array.  Whole-program type inference emits this representation
 * for ListInt values, cutting list memory traffic in half and letting LLVM see
 * plain integer loads/stores. */
typedef struct SkIntList {
    int64_t len;
    int64_t cap;
    int64_t *items;
} SkIntList;

typedef struct SkEntry {
    SkValue key;
    SkValue value;
} SkEntry;

typedef struct SkDict {
    int64_t len;
    int64_t cap;
    SkEntry *items;
} SkDict;

typedef struct SkRange {
    int64_t start;
    int64_t stop;
    int64_t step;
} SkRange;

typedef struct SkClosure SkClosure;
typedef SkValue (*SkCode)(SkClosure *, int64_t, SkValue *, int64_t, SkValue *, SkValue *);

struct SkClosure {
    SkCode code;
    int64_t nenv;
    SkValue **env;
};

typedef struct SkMethod {
    SkValue recv;
    int64_t id;
} SkMethod;

typedef struct SkCallArgs {
    int64_t nargs;
    int64_t cap;
    SkValue *args;
    int64_t nkwargs;
    int64_t kwcap;
    SkValue *kw_names;
    SkValue *kw_values;
} SkCallArgs;

static void sk_fail(const char *msg);
static void sk_failf(const char *fmt, ...);
static int value_equal(SkValue a, SkValue b);
static int value_compare(SkValue a, SkValue b);
static SkValue value_repr(SkValue v);
static SkValue value_str(SkValue v);
static char *value_to_cstr(SkValue v);
static int64_t range_len(SkRange *r);
static SkValue sk_iter_at_impl(SkValue v, int64_t i);
static int64_t sk_iter_len_impl(SkValue v);
static SkValue sk_dict_get_impl(SkValue dict, SkValue key, int required, SkValue def);
static void *xmalloc(size_t n);
static void *xrealloc(void *p, size_t n);
SkValue sk_call(SkValue callee, SkCallArgs *a);

/* Large list payloads dominate the hash workloads.  Back them with
 * transparent huge pages so the kernel faults in one 2 MiB page instead of
 * 512 individual 4 KiB pages.  Rather than rounding every allocation up to a
 * whole huge page (which wastes and, more importantly, zeroes up to 2 MiB per
 * 512 KiB request), carve large allocations out of shared 2 MiB-aligned
 * chunks.  The runtime is leak-friendly, so the memory is reclaimed at
 * process exit. */
#define SK_HUGE_ALIGN (2 * 1024 * 1024)
#define SK_HUGE_CHUNK (16 * 1024 * 1024)
#define SK_HUGE_MIN (256 * 1024)

typedef struct SkHugeChunk {
    struct SkHugeChunk *next;
    char *base;
    size_t cap;
} SkHugeChunk;

typedef struct SkHugeHdr {
    size_t size;
    size_t pad;
} SkHugeHdr;

static SkHugeChunk *sk_huge_chunks;
static char *sk_huge_cur;
static size_t sk_huge_left;

static int sk_is_huge(const void *p) {
    for (SkHugeChunk *c = sk_huge_chunks; c; c = c->next) {
        const char *q = (const char *)p;
        if (q >= c->base && q < c->base + c->cap) return 1;
    }
    return 0;
}

static void sk_record_chunk(char *base, size_t cap) {
    SkHugeChunk *c = (SkHugeChunk *)malloc(sizeof(SkHugeChunk));
    if (!c) return;
    c->base = base;
    c->cap = cap;
    c->next = sk_huge_chunks;
    sk_huge_chunks = c;
}

static void *sk_huge_alloc(size_t n) {
    size_t need = (n + sizeof(SkHugeHdr) + 15) & ~(size_t)15;
    char *base;
    if (need > SK_HUGE_CHUNK / 2) {
        size_t rounded = (need + SK_HUGE_ALIGN - 1) & ~(size_t)(SK_HUGE_ALIGN - 1);
        void *p = NULL;
        if (posix_memalign(&p, SK_HUGE_ALIGN, rounded) != 0) return NULL;
        madvise(p, rounded, MADV_HUGEPAGE);
        sk_record_chunk((char *)p, rounded);
        base = (char *)p;
    } else {
        if (need > sk_huge_left) {
            void *p = NULL;
            if (posix_memalign(&p, SK_HUGE_ALIGN, SK_HUGE_CHUNK) != 0) return NULL;
            madvise(p, SK_HUGE_CHUNK, MADV_HUGEPAGE);
            sk_record_chunk((char *)p, SK_HUGE_CHUNK);
            sk_huge_cur = (char *)p;
            sk_huge_left = SK_HUGE_CHUNK;
        }
        base = sk_huge_cur;
        sk_huge_cur += need;
        sk_huge_left -= need;
    }
    SkHugeHdr *h = (SkHugeHdr *)base;
    h->size = n;
    return h + 1;
}

static void *xmalloc(size_t n) {
    if (!n) n = 1;
    if (n >= SK_HUGE_MIN) {
        void *p = sk_huge_alloc(n);
        if (p) return p;
    }
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "Starlark error: out of memory\n");
        exit(1);
    }
    return p;
}

static void *xrealloc(void *p, size_t n) {
    if (!p) return xmalloc(n);
    if (sk_is_huge(p)) {
        SkHugeHdr *h = (SkHugeHdr *)p - 1;
        void *q = xmalloc(n);
        size_t old = h->size;
        memcpy(q, p, old < n ? old : n);
        return q;
    }
    void *q = realloc(p, n ? n : 1);
    if (!q) {
        fprintf(stderr, "Starlark error: out of memory\n");
        exit(1);
    }
    return q;
}

static void xfree(void *p) {
    if (p && !sk_is_huge(p)) free(p);
}

static void sk_fail(const char *msg) {
    fprintf(stderr, "Starlark error: %s\n", msg);
    exit(1);
}

static void sk_failf(const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "Starlark error: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    exit(1);
}

static SkValue v_none(void) {
    return (SkValue){SK_NONE, 0};
}

static SkValue v_bool(int b) {
    return (SkValue){SK_BOOL, b ? 1 : 0};
}

static SkValue v_int(int64_t i) {
    return (SkValue){SK_INT, i};
}

static SkValue v_float(double d) {
    union {
        double d;
        int64_t i;
    } u;
    u.d = d;
    return (SkValue){SK_FLOAT, u.i};
}

static double as_float(SkValue v) {
    union {
        double d;
        int64_t i;
    } u;
    u.i = v.payload;
    return u.d;
}

static SkString *as_string(SkValue v) {
    return (SkString *)(intptr_t)v.payload;
}

static SkList *as_list(SkValue v) {
    return (SkList *)(intptr_t)v.payload;
}

static SkIntList *as_int_list(SkValue v) {
    return (SkIntList *)(intptr_t)v.payload;
}

static SkDict *as_dict(SkValue v) {
    return (SkDict *)(intptr_t)v.payload;
}

static SkRange *as_range(SkValue v) {
    return (SkRange *)(intptr_t)v.payload;
}

static SkClosure *as_closure(SkValue v) {
    return (SkClosure *)(intptr_t)v.payload;
}

static SkMethod *as_method(SkValue v) {
    return (SkMethod *)(intptr_t)v.payload;
}

static int is_number(SkValue v) {
    return v.tag == SK_INT || v.tag == SK_FLOAT;
}

static int is_sequence(SkValue v) {
    return v.tag == SK_LIST || v.tag == SK_INT_LIST || v.tag == SK_TUPLE ||
           v.tag == SK_STRING_ELEMS;
}

static int is_hashable(SkValue v) {
    switch (v.tag) {
    case SK_NONE:
    case SK_BOOL:
    case SK_INT:
    case SK_FLOAT:
    case SK_STRING:
    case SK_FUNCTION:
    case SK_BUILTIN:
    case SK_RANGE:
    case SK_CHAR:
        return 1;
    case SK_TUPLE: {
        SkList *l = as_list(v);
        for (int64_t i = 0; i < l->len; i++) {
            if (!is_hashable(l->items[i])) return 0;
        }
        return 1;
    }
    default:
        return 0;
    }
}

/* ------------------------------------------------------------------------- */
/* Constructors and cells                                                    */
/* ------------------------------------------------------------------------- */

void sk_runtime_init(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
}

SkValue sk_none(void) {
    return v_none();
}

SkValue sk_bool(int64_t b) {
    return v_bool(b != 0);
}

SkValue sk_int(int64_t i) {
    return v_int(i);
}

SkValue sk_float(double d) {
    return v_float(d);
}

SkValue sk_unbound(void) {
    return (SkValue){SK_UNBOUND, 0};
}

SkValue sk_string(const char *data, int64_t len) {
    if (len < 0) sk_fail("negative string length");
    SkString *s = (SkString *)xmalloc(sizeof(SkString) + (size_t)len + 1);
    s->len = len;
    if (len) memcpy(s->data, data, (size_t)len);
    s->data[len] = '\0';
    return (SkValue){SK_STRING, (int64_t)(intptr_t)s};
}

static SkValue sk_string_cstr(const char *s) {
    return sk_string(s, (int64_t)strlen(s));
}

SkValue *sk_cell_new(SkValue v) {
    SkValue *p = (SkValue *)xmalloc(sizeof(SkValue));
    *p = v;
    return p;
}

SkValue **sk_env_new(int64_t n) {
    if (n < 0) sk_fail("negative environment size");
    SkValue **env = (SkValue **)xmalloc(sizeof(SkValue *) * (size_t)(n ? n : 1));
    for (int64_t i = 0; i < n; i++) env[i] = NULL;
    return env;
}

void sk_env_set(SkValue **env, int64_t i, SkValue *cell) {
    env[i] = cell;
}

SkValue sk_function_new(SkCode code, int64_t nenv, SkValue **env) {
    SkClosure *cl = (SkClosure *)xmalloc(sizeof(SkClosure));
    cl->code = code;
    cl->nenv = nenv;
    cl->env = env;
    return (SkValue){SK_FUNCTION, (int64_t)(intptr_t)cl};
}

SkValue sk_builtin(int64_t id) {
    return (SkValue){SK_BUILTIN, id};
}

SkValue sk_check_bound(SkValue v, const char *name, int64_t len) {
    if (v.tag == SK_UNBOUND) {
        sk_failf("variable '%.*s' referenced before assignment", (int)len, name);
    }
    return v;
}

SkValue sk_undefined(const char *name, int64_t len) {
    sk_failf("name '%.*s' is not defined", (int)len, name);
    return v_none();
}

void sk_error_cstr(const char *msg, int64_t len) {
    sk_failf("%.*s", (int)len, msg);
}

/* ------------------------------------------------------------------------- */
/* Lists, tuples, dictionaries, ranges                                       */
/* ------------------------------------------------------------------------- */

static void list_reserve(SkList *l, int64_t need) {
    if (need <= l->cap) return;
    int64_t cap = l->cap ? l->cap : 4;
    while (cap < need) {
        if (cap > INT64_MAX / 2) {
            cap = need;
            break;
        }
        cap *= 2;
    }
    l->items = (SkValue *)xrealloc(l->items, sizeof(SkValue) * (size_t)cap);
    l->cap = cap;
}

static void int_list_reserve(SkIntList *l, int64_t need) {
    if (need <= l->cap) return;
    int64_t cap = l->cap ? l->cap : 8;
    while (cap < need) {
        if (cap > INT64_MAX / 2) {
            cap = need;
            break;
        }
        cap *= 2;
    }
    l->items = (int64_t *)xrealloc(l->items, sizeof(int64_t) * (size_t)cap);
    l->cap = cap;
}

static SkValue int_list_new_tagged(void) {
    SkIntList *l = (SkIntList *)xmalloc(sizeof(SkIntList));
    l->len = 0;
    l->cap = 0;
    l->items = NULL;
    return (SkValue){SK_INT_LIST, (int64_t)(intptr_t)l};
}

SkValue sk_int_list_new(void) {
    return int_list_new_tagged();
}

SkValue sk_int_list_new_cap(int64_t cap) {
    if (cap < 0) sk_fail("negative list capacity");
    SkValue v = int_list_new_tagged();
    if (cap) int_list_reserve(as_int_list(v), cap);
    return v;
}

void sk_int_list_reserve(SkValue list, int64_t cap) {
    if (list.tag != SK_INT_LIST) sk_fail("reserve on non-int-list");
    if (cap < 0) sk_fail("negative list capacity");
    int_list_reserve(as_int_list(list), cap);
}

SkValue sk_int_list_zeros(int64_t n) {
    if (n < 0) sk_fail("negative list length");
    SkValue v = int_list_new_tagged();
    SkIntList *l = as_int_list(v);
    int_list_reserve(l, n);
    if (n) memset(l->items, 0, sizeof(int64_t) * (size_t)n);
    l->len = n;
    return v;
}

SkValue sk_int_list_new_from(int64_t n, int64_t *items) {
    if (n < 0) sk_fail("negative list length");
    SkValue v = int_list_new_tagged();
    SkIntList *l = as_int_list(v);
    int_list_reserve(l, n);
    if (n && items) memcpy(l->items, items, sizeof(int64_t) * (size_t)n);
    l->len = n;
    return v;
}

/* Concatenate several compact integer lists into one allocation.  The
 * compiler uses this for chained `a + b + c` expressions, which are common in
 * padding and byte-packing code.  The generic `sk_add` path would allocate and
 * copy an intermediate list for every `+`. */
SkValue sk_int_list_concat(int64_t nparts, SkValue *parts) {
    int64_t total = 0;
    for (int64_t i = 0; i < nparts; i++) {
        if (parts[i].tag == SK_INT_LIST) {
            SkIntList *p = as_int_list(parts[i]);
            if (__builtin_add_overflow(total, p->len, &total)) sk_fail("list too large");
        } else if (parts[i].tag == SK_LIST) {
            SkList *p = as_list(parts[i]);
            for (int64_t j = 0; j < p->len; j++) {
                if (p->items[j].tag != SK_INT) sk_fail("int list concatenation expected");
            }
            if (__builtin_add_overflow(total, p->len, &total)) sk_fail("list too large");
        } else {
            sk_fail("int list concatenation expected");
        }
    }
    SkValue out = sk_int_list_new_from(total, NULL);
    SkIntList *dst = as_int_list(out);
    int64_t at = 0;
    for (int64_t i = 0; i < nparts; i++) {
        if (parts[i].tag == SK_INT_LIST) {
            SkIntList *p = as_int_list(parts[i]);
            if (p->len) {
                memcpy(dst->items + at, p->items, sizeof(int64_t) * (size_t)p->len);
                at += p->len;
            }
        } else {
            SkList *p = as_list(parts[i]);
            for (int64_t j = 0; j < p->len; j++) dst->items[at++] = p->items[j].payload;
        }
    }
    return out;
}

void sk_int_list_push(SkValue list, int64_t item) {
    if (list.tag != SK_INT_LIST) sk_fail("append on non-int-list");
    SkIntList *l = as_int_list(list);
    int_list_reserve(l, l->len + 1);
    l->items[l->len++] = item;
}

static SkValue list_new_tagged(int64_t tag) {
    SkList *l = (SkList *)xmalloc(sizeof(SkList));
    l->len = 0;
    l->cap = 0;
    l->items = NULL;
    return (SkValue){tag, (int64_t)(intptr_t)l};
}

SkValue sk_list_new(void) {
    return list_new_tagged(SK_LIST);
}

SkValue sk_list_new_from(int64_t n, SkValue *items) {
    if (n < 0) sk_fail("negative list length");
    SkValue v = list_new_tagged(SK_LIST);
    SkList *l = as_list(v);
    list_reserve(l, n);
    if (n && items) memcpy(l->items, items, sizeof(SkValue) * (size_t)n);
    l->len = n;
    return v;
}

SkValue sk_tuple_new(int64_t n, SkValue *items) {
    if (n < 0) sk_fail("negative tuple length");
    SkValue v = list_new_tagged(SK_TUPLE);
    SkList *l = as_list(v);
    list_reserve(l, n);
    if (n && items) memcpy(l->items, items, sizeof(SkValue) * (size_t)n);
    l->len = n;
    return v;
}

void sk_list_push(SkValue list, SkValue item) {
    if (list.tag == SK_INT_LIST) {
        if (item.tag != SK_INT) sk_fail("int list element expected");
        sk_int_list_push(list, item.payload);
        return;
    }
    if (list.tag != SK_LIST) sk_fail("append on non-list");
    SkList *l = as_list(list);
    list_reserve(l, l->len + 1);
    l->items[l->len++] = item;
}

SkValue sk_tuple_slice(SkValue *args, int64_t nargs, int64_t start) {
    if (start < 0) start = 0;
    if (start > nargs) start = nargs;
    return sk_tuple_new(nargs - start, args ? args + start : NULL);
}

SkValue sk_dict_new(void) {
    SkDict *d = (SkDict *)xmalloc(sizeof(SkDict));
    d->len = 0;
    d->cap = 0;
    d->items = NULL;
    return (SkValue){SK_DICT, (int64_t)(intptr_t)d};
}

SkValue sk_struct_new(void) {
    SkValue v = sk_dict_new();
    v.tag = SK_STRUCT;
    return v;
}

static void dict_reserve(SkDict *d, int64_t need) {
    if (need <= d->cap) return;
    int64_t cap = d->cap ? d->cap : 4;
    while (cap < need) {
        if (cap > INT64_MAX / 2) {
            cap = need;
            break;
        }
        cap *= 2;
    }
    d->items = (SkEntry *)xrealloc(d->items, sizeof(SkEntry) * (size_t)cap);
    d->cap = cap;
}

static int64_t dict_find(SkDict *d, SkValue key) {
    for (int64_t i = 0; i < d->len; i++) {
        if (value_equal(d->items[i].key, key)) return i;
    }
    return -1;
}

void sk_dict_set(SkValue dict, SkValue key, SkValue value) {
    if (dict.tag != SK_DICT && dict.tag != SK_STRUCT) sk_fail("dictionary update on non-dict");
    if (!is_hashable(key)) sk_fail("unhashable dictionary key");
    SkDict *d = as_dict(dict);
    int64_t i = dict_find(d, key);
    if (i >= 0) {
        d->items[i].value = value;
        return;
    }
    dict_reserve(d, d->len + 1);
    d->items[d->len].key = key;
    d->items[d->len].value = value;
    d->len++;
}

static SkValue sk_dict_get_impl(SkValue dict, SkValue key, int required, SkValue def) {
    if (dict.tag != SK_DICT) sk_fail("dictionary lookup on non-dict");
    SkDict *d = as_dict(dict);
    int64_t i = dict_find(d, key);
    if (i < 0) {
        if (required) sk_failf("key not found in dictionary");
        return def;
    }
    return d->items[i].value;
}

static void dict_remove_at(SkDict *d, int64_t i) {
    if (i < 0 || i >= d->len) return;
    for (int64_t j = i + 1; j < d->len; j++) d->items[j - 1] = d->items[j];
    d->len--;
}

SkValue sk_range(int64_t start, int64_t stop, int64_t step) {
    if (step == 0) sk_fail("range step cannot be zero");
    SkRange *r = (SkRange *)xmalloc(sizeof(SkRange));
    r->start = start;
    r->stop = stop;
    r->step = step;
    return (SkValue){SK_RANGE, (int64_t)(intptr_t)r};
}

static int64_t range_len(SkRange *r) {
    __int128 start = r->start;
    __int128 stop = r->stop;
    __int128 step = r->step;
    __int128 n;
    if (step > 0) {
        if (start >= stop) return 0;
        n = (stop - start + step - 1) / step;
    } else {
        if (start <= stop) return 0;
        n = (start - stop + (-step) - 1) / (-step);
    }
    if (n > INT64_MAX) sk_fail("range is too large");
    return (int64_t)n;
}

int64_t sk_range_len_i64(int64_t start, int64_t stop, int64_t step) {
    if (step == 0) sk_fail("range step cannot be zero");
    __int128 s = start;
    __int128 e = stop;
    __int128 d = step;
    __int128 n;
    if (d > 0) {
        if (s >= e) return 0;
        n = (e - s + d - 1) / d;
    } else {
        if (s <= e) return 0;
        n = (s - e + (-d) - 1) / (-d);
    }
    if (n > INT64_MAX) sk_fail("range is too large");
    return (int64_t)n;
}

/* ------------------------------------------------------------------------- */
/* Iteration and truthiness                                                  */
/* ------------------------------------------------------------------------- */

static int64_t sk_iter_len_impl(SkValue v) {
    switch (v.tag) {
    case SK_LIST:
    case SK_TUPLE:
        return as_list(v)->len;
    case SK_INT_LIST:
        return as_int_list(v)->len;
    case SK_STRING:
    case SK_STRING_ELEMS:
        return as_string(v)->len;
    case SK_DICT:
        return as_dict(v)->len;
    case SK_RANGE:
        return range_len(as_range(v));
    default:
        sk_fail("value is not iterable");
        return 0;
    }
}

int64_t sk_iter_len(SkValue v) {
    return sk_iter_len_impl(v);
}

static SkValue sk_iter_at_impl(SkValue v, int64_t i) {
    switch (v.tag) {
    case SK_LIST:
    case SK_TUPLE: {
        SkList *l = as_list(v);
        if (i < 0 || i >= l->len) sk_fail("iteration index out of range");
        return l->items[i];
    }
    case SK_INT_LIST: {
        SkIntList *l = as_int_list(v);
        if (i < 0 || i >= l->len) sk_fail("iteration index out of range");
        return v_int(l->items[i]);
    }
    case SK_STRING: {
        SkString *s = as_string(v);
        if (i < 0 || i >= s->len) sk_fail("iteration index out of range");
        return sk_string(s->data + i, 1);
    }
    case SK_STRING_ELEMS: {
        SkString *s = as_string(v);
        if (i < 0 || i >= s->len) sk_fail("iteration index out of range");
        return (SkValue){SK_CHAR, (unsigned char)s->data[i]};
    }
    case SK_DICT: {
        SkDict *d = as_dict(v);
        if (i < 0 || i >= d->len) sk_fail("iteration index out of range");
        return d->items[i].key;
    }
    case SK_RANGE: {
        SkRange *r = as_range(v);
        int64_t n = range_len(r);
        if (i < 0 || i >= n) sk_fail("iteration index out of range");
        return v_int(r->start + i * r->step);
    }
    default:
        sk_fail("value is not iterable");
        return v_none();
    }
}

SkValue sk_iter_at(SkValue v, int64_t i) {
    return sk_iter_at_impl(v, i);
}

void sk_unpack_len(SkValue v, int64_t expected) {
    int64_t n = sk_iter_len_impl(v);
    if (n != expected) {
        sk_failf("cannot unpack %" PRId64 " values into %" PRId64 " targets", n, expected);
    }
}

int64_t sk_truthy(SkValue v) {
    switch (v.tag) {
    case SK_NONE:
    case SK_UNBOUND:
        return 0;
    case SK_BOOL:
    case SK_INT:
        return v.payload != 0;
    case SK_FLOAT:
        return as_float(v) != 0.0;
    case SK_STRING:
        return as_string(v)->len != 0;
    case SK_LIST:
    case SK_TUPLE:
        return as_list(v)->len != 0;
    case SK_INT_LIST:
        return as_int_list(v)->len != 0;
    case SK_STRING_ELEMS:
        return as_string(v)->len != 0;
    case SK_CHAR:
        return v.payload != 0;
    case SK_DICT:
        return as_dict(v)->len != 0;
    case SK_RANGE:
        return range_len(as_range(v)) != 0;
    default:
        return 1;
    }
}

/* ------------------------------------------------------------------------- */
/* Equality and comparison                                                   */
/* ------------------------------------------------------------------------- */

static int value_equal(SkValue a, SkValue b) {
    if (a.tag == b.tag) {
        switch (a.tag) {
        case SK_NONE:
        case SK_UNBOUND:
            return 1;
        case SK_BOOL:
        case SK_INT:
        case SK_BUILTIN:
        case SK_CHAR:
            return a.payload == b.payload;
        case SK_FLOAT:
            return as_float(a) == as_float(b);
        case SK_STRING: {
            SkString *x = as_string(a), *y = as_string(b);
            return x->len == y->len && memcmp(x->data, y->data, (size_t)x->len) == 0;
        }
        case SK_INT_LIST: {
            SkIntList *x = as_int_list(a), *y = as_int_list(b);
            if (x->len != y->len) return 0;
            return x->len == 0 || memcmp(x->items, y->items, sizeof(int64_t) * (size_t)x->len) == 0;
        }
        case SK_LIST:
        case SK_TUPLE: {
            SkList *x = as_list(a), *y = as_list(b);
            if (x->len != y->len) return 0;
            for (int64_t i = 0; i < x->len; i++) {
                if (!value_equal(x->items[i], y->items[i])) return 0;
            }
            return 1;
        }
        case SK_DICT:
        case SK_STRUCT: {
            SkDict *x = as_dict(a), *y = as_dict(b);
            if (x->len != y->len) return 0;
            for (int64_t i = 0; i < x->len; i++) {
                int64_t j = dict_find(y, x->items[i].key);
                if (j < 0 || !value_equal(x->items[i].value, y->items[j].value)) return 0;
            }
            return 1;
        }
        case SK_RANGE: {
            SkRange *x = as_range(a), *y = as_range(b);
            return x->start == y->start && x->stop == y->stop && x->step == y->step;
        }
        case SK_FUNCTION:
        case SK_METHOD:
            return a.payload == b.payload;
        default:
            return a.payload == b.payload;
        }
    }
    if (is_number(a) && is_number(b)) {
        if (a.tag == SK_INT && b.tag == SK_INT) return a.payload == b.payload;
        double x = a.tag == SK_INT ? (double)a.payload : as_float(a);
        double y = b.tag == SK_INT ? (double)b.payload : as_float(b);
        return x == y;
    }
    return 0;
}

static int value_compare(SkValue a, SkValue b) {
    if (is_number(a) && is_number(b)) {
        if (a.tag == SK_INT && b.tag == SK_INT) {
            return a.payload < b.payload ? -1 : a.payload > b.payload ? 1 : 0;
        }
        double x = a.tag == SK_INT ? (double)a.payload : as_float(a);
        double y = b.tag == SK_INT ? (double)b.payload : as_float(b);
        return x < y ? -1 : x > y ? 1 : 0;
    }
    if (a.tag == SK_STRING && b.tag == SK_STRING) {
        SkString *x = as_string(a), *y = as_string(b);
        int64_t n = x->len < y->len ? x->len : y->len;
        int c = memcmp(x->data, y->data, (size_t)n);
        if (c) return c < 0 ? -1 : 1;
        return x->len < y->len ? -1 : x->len > y->len ? 1 : 0;
    }
    if (a.tag == SK_BOOL && b.tag == SK_BOOL) {
        return a.payload < b.payload ? -1 : a.payload > b.payload ? 1 : 0;
    }
    if (a.tag == SK_INT_LIST && b.tag == SK_INT_LIST) {
        SkIntList *x = as_int_list(a), *y = as_int_list(b);
        int64_t n = x->len < y->len ? x->len : y->len;
        for (int64_t i = 0; i < n; i++) {
            int c = x->items[i] < y->items[i] ? -1 : x->items[i] > y->items[i] ? 1 : 0;
            if (c) return c;
        }
        return x->len < y->len ? -1 : x->len > y->len ? 1 : 0;
    }
    if (is_sequence(a) && is_sequence(b) && a.tag == b.tag) {
        SkList *x = as_list(a), *y = as_list(b);
        int64_t n = x->len < y->len ? x->len : y->len;
        for (int64_t i = 0; i < n; i++) {
            int c = value_compare(x->items[i], y->items[i]);
            if (c) return c;
        }
        return x->len < y->len ? -1 : x->len > y->len ? 1 : 0;
    }
    if (a.tag == SK_RANGE && b.tag == SK_RANGE) {
        SkRange *x = as_range(a), *y = as_range(b);
        if (x->start != y->start) return x->start < y->start ? -1 : 1;
        if (x->stop != y->stop) return x->stop < y->stop ? -1 : 1;
        if (x->step != y->step) return x->step < y->step ? -1 : 1;
        return 0;
    }
    sk_fail("values are not comparable");
    return 0;
}

SkValue sk_eq(SkValue a, SkValue b) {
    return v_bool(value_equal(a, b));
}

SkValue sk_ne(SkValue a, SkValue b) {
    return v_bool(!value_equal(a, b));
}

SkValue sk_lt(SkValue a, SkValue b) {
    return v_bool(value_compare(a, b) < 0);
}

SkValue sk_gt(SkValue a, SkValue b) {
    return v_bool(value_compare(a, b) > 0);
}

SkValue sk_le(SkValue a, SkValue b) {
    return v_bool(value_compare(a, b) <= 0);
}

SkValue sk_ge(SkValue a, SkValue b) {
    return v_bool(value_compare(a, b) >= 0);
}

/* ------------------------------------------------------------------------- */
/* Arithmetic                                                                */
/* ------------------------------------------------------------------------- */

/* Starlark integers are arbitrary precision. This frontend represents them as
 * 64-bit words and uses two's-complement wrapping, which is what the pure
 * Starlark hash implementations in hashes.bzl expect (they mask every result
 * back to 32 or 64 bits themselves). */
static int64_t checked_add(int64_t a, int64_t b) {
    return (int64_t)((uint64_t)a + (uint64_t)b);
}

static int64_t checked_sub(int64_t a, int64_t b) {
    return (int64_t)((uint64_t)a - (uint64_t)b);
}

static int64_t checked_mul(int64_t a, int64_t b) {
    return (int64_t)((uint64_t)a * (uint64_t)b);
}

static SkValue string_concat(SkValue a, SkValue b) {
    SkString *x = as_string(a), *y = as_string(b);
    SkString *s = (SkString *)xmalloc(sizeof(SkString) + (size_t)(x->len + y->len) + 1);
    s->len = x->len + y->len;
    memcpy(s->data, x->data, (size_t)x->len);
    memcpy(s->data + x->len, y->data, (size_t)y->len);
    s->data[s->len] = '\0';
    return (SkValue){SK_STRING, (int64_t)(intptr_t)s};
}

static SkValue int_list_to_list(SkValue v) {
    SkIntList *src = as_int_list(v);
    SkValue out = sk_list_new_from(src->len, NULL);
    SkList *dst = as_list(out);
    for (int64_t i = 0; i < src->len; i++) dst->items[i] = v_int(src->items[i]);
    return out;
}

SkValue sk_add(SkValue a, SkValue b) {
    if (a.tag == SK_INT && b.tag == SK_INT) return v_int(checked_add(a.payload, b.payload));
    if (is_number(a) && is_number(b)) {
        double x = a.tag == SK_INT ? (double)a.payload : as_float(a);
        double y = b.tag == SK_INT ? (double)b.payload : as_float(b);
        return v_float(x + y);
    }
    if (a.tag == SK_STRING && b.tag == SK_STRING) return string_concat(a, b);
    if (a.tag == SK_INT_LIST && b.tag == SK_INT_LIST) {
        SkIntList *x = as_int_list(a), *y = as_int_list(b);
        SkValue out = sk_int_list_new_from(x->len + y->len, NULL);
        SkIntList *l = as_int_list(out);
        memcpy(l->items, x->items, sizeof(int64_t) * (size_t)x->len);
        memcpy(l->items + x->len, y->items, sizeof(int64_t) * (size_t)y->len);
        return out;
    }
    if ((a.tag == SK_INT_LIST || b.tag == SK_INT_LIST) &&
        (a.tag == SK_LIST || a.tag == SK_INT_LIST) &&
        (b.tag == SK_LIST || b.tag == SK_INT_LIST)) {
        SkValue ga = a.tag == SK_INT_LIST ? int_list_to_list(a) : a;
        SkValue gb = b.tag == SK_INT_LIST ? int_list_to_list(b) : b;
        SkList *x = as_list(ga), *y = as_list(gb);
        SkValue out = sk_list_new_from(x->len + y->len, NULL);
        SkList *l = as_list(out);
        memcpy(l->items, x->items, sizeof(SkValue) * (size_t)x->len);
        memcpy(l->items + x->len, y->items, sizeof(SkValue) * (size_t)y->len);
        return out;
    }
    if (a.tag == SK_LIST && b.tag == SK_LIST) {
        SkList *x = as_list(a), *y = as_list(b);
        SkValue out = sk_list_new_from(x->len + y->len, NULL);
        SkList *l = as_list(out);
        memcpy(l->items, x->items, sizeof(SkValue) * (size_t)x->len);
        memcpy(l->items + x->len, y->items, sizeof(SkValue) * (size_t)y->len);
        return out;
    }
    if (a.tag == SK_TUPLE && b.tag == SK_TUPLE) {
        SkList *x = as_list(a), *y = as_list(b);
        SkValue out = sk_tuple_new(x->len + y->len, NULL);
        SkList *l = as_list(out);
        memcpy(l->items, x->items, sizeof(SkValue) * (size_t)x->len);
        memcpy(l->items + x->len, y->items, sizeof(SkValue) * (size_t)y->len);
        return out;
    }
    sk_fail("unsupported operands for +");
    return v_none();
}

SkValue sk_sub(SkValue a, SkValue b) {
    if (a.tag == SK_INT && b.tag == SK_INT) return v_int(checked_sub(a.payload, b.payload));
    if (is_number(a) && is_number(b)) {
        double x = a.tag == SK_INT ? (double)a.payload : as_float(a);
        double y = b.tag == SK_INT ? (double)b.payload : as_float(b);
        return v_float(x - y);
    }
    sk_fail("unsupported operands for -");
    return v_none();
}

static SkValue repeat_sequence(SkValue seq, int64_t n) {
    if (n < 0) n = 0;
    if (seq.tag == SK_INT_LIST) {
        SkIntList *x = as_int_list(seq);
        int64_t total;
        if (__builtin_mul_overflow(x->len, n, &total)) sk_fail("sequence too large");
        SkValue out = sk_int_list_new_from(total, NULL);
        SkIntList *l = as_int_list(out);
        for (int64_t i = 0; i < n; i++) {
            memcpy(l->items + i * x->len, x->items, sizeof(int64_t) * (size_t)x->len);
        }
        l->len = total;
        return out;
    }
    SkList *x = as_list(seq);
    int64_t total;
    if (__builtin_mul_overflow(x->len, n, &total)) sk_fail("sequence too large");
    SkValue out = seq.tag == SK_LIST ? sk_list_new_from(total, NULL) : sk_tuple_new(total, NULL);
    SkList *l = as_list(out);
    for (int64_t i = 0; i < n; i++) {
        memcpy(l->items + i * x->len, x->items, sizeof(SkValue) * (size_t)x->len);
    }
    l->len = total;
    return out;
}

SkValue sk_mul(SkValue a, SkValue b) {
    if (a.tag == SK_INT && b.tag == SK_INT) return v_int(checked_mul(a.payload, b.payload));
    if (is_number(a) && is_number(b)) {
        double x = a.tag == SK_INT ? (double)a.payload : as_float(a);
        double y = b.tag == SK_INT ? (double)b.payload : as_float(b);
        return v_float(x * y);
    }
    if (a.tag == SK_STRING && b.tag == SK_INT) {
        if (b.payload <= 0) return sk_string("", 0);
        SkString *s = as_string(a);
        int64_t total;
        if (__builtin_mul_overflow(s->len, b.payload, &total)) sk_fail("string too large");
        char *buf = (char *)xmalloc((size_t)total + 1);
        for (int64_t i = 0; i < b.payload; i++) memcpy(buf + i * s->len, s->data, (size_t)s->len);
        buf[total] = '\0';
        SkValue out = sk_string(buf, total);
        xfree(buf);
        return out;
    }
    if (b.tag == SK_STRING && a.tag == SK_INT) return sk_mul(b, a);
    if (is_sequence(a) && b.tag == SK_INT) return repeat_sequence(a, b.payload);
    if (is_sequence(b) && a.tag == SK_INT) return repeat_sequence(b, a.payload);
    sk_fail("unsupported operands for *");
    return v_none();
}

SkValue sk_div(SkValue a, SkValue b) {
    if (!is_number(a) || !is_number(b)) sk_fail("unsupported operands for /");
    double x = a.tag == SK_INT ? (double)a.payload : as_float(a);
    double y = b.tag == SK_INT ? (double)b.payload : as_float(b);
    if (y == 0.0) sk_fail("division by zero");
    return v_float(x / y);
}

SkValue sk_floordiv(SkValue a, SkValue b) {
    if (a.tag == SK_INT && b.tag == SK_INT) {
        if (b.payload == 0) sk_fail("division by zero");
        int64_t q = a.payload / b.payload;
        int64_t r = a.payload % b.payload;
        if (r != 0 && ((r < 0) != (b.payload < 0))) q--;
        return v_int(q);
    }
    if (is_number(a) && is_number(b)) {
        double x = a.tag == SK_INT ? (double)a.payload : as_float(a);
        double y = b.tag == SK_INT ? (double)b.payload : as_float(b);
        if (y == 0.0) sk_fail("division by zero");
        return v_float(floor(x / y));
    }
    sk_fail("unsupported operands for //");
    return v_none();
}

static SkValue format_percent(SkValue fmt, SkValue arg);

SkValue sk_mod(SkValue a, SkValue b) {
    if (a.tag == SK_STRING) return format_percent(a, b);
    if (a.tag == SK_INT && b.tag == SK_INT) {
        if (b.payload == 0) sk_fail("modulo by zero");
        int64_t r = a.payload % b.payload;
        if (r != 0 && ((r < 0) != (b.payload < 0))) r += b.payload;
        return v_int(r);
    }
    if (is_number(a) && is_number(b)) {
        double x = a.tag == SK_INT ? (double)a.payload : as_float(a);
        double y = b.tag == SK_INT ? (double)b.payload : as_float(b);
        if (y == 0.0) sk_fail("modulo by zero");
        double r = fmod(x, y);
        if (r != 0.0 && ((r < 0) != (y < 0))) r += y;
        return v_float(r);
    }
    sk_fail("unsupported operands for %");
    return v_none();
}

static int64_t need_int(SkValue v, const char *op) {
    if (v.tag != SK_INT) sk_failf("unsupported operand for %s", op);
    return v.payload;
}

SkValue sk_bitand(SkValue a, SkValue b) {
    return v_int(need_int(a, "&") & need_int(b, "&"));
}

SkValue sk_bitor(SkValue a, SkValue b) {
    return v_int(need_int(a, "|") | need_int(b, "|"));
}

SkValue sk_bitxor(SkValue a, SkValue b) {
    return v_int(need_int(a, "^") ^ need_int(b, "^"));
}

SkValue sk_shl(SkValue a, SkValue b) {
    int64_t x = need_int(a, "<<");
    int64_t n = need_int(b, "<<");
    if (n < 0) sk_fail("negative shift count");
    if (n >= 64) return v_int(0);
    return v_int((int64_t)((uint64_t)x << (uint64_t)n));
}

SkValue sk_shr(SkValue a, SkValue b) {
    int64_t x = need_int(a, ">>");
    int64_t n = need_int(b, ">>");
    if (n < 0) sk_fail("negative shift count");
    /* Values above INT64_MAX are represented as negative i64s, so use a
     * logical shift to recover Starlark's non-negative integer semantics. */
    if (n >= 64) return v_int(0);
    return v_int((int64_t)((uint64_t)x >> (uint64_t)n));
}

int64_t sk_floordiv_i64(int64_t a, int64_t b) {
    if (b == 0) sk_fail("integer division by zero");
    int64_t q = a / b;
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) q -= 1;
    return q;
}

int64_t sk_mod_i64(int64_t a, int64_t b) {
    if (b == 0) sk_fail("integer modulo by zero");
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}

int64_t sk_ord_int(SkValue v) {
    if (v.tag == SK_CHAR) return v.payload;
    if (v.tag != SK_STRING || as_string(v)->len != 1) {
        sk_fail("ord() expects a one-character string");
    }
    return (unsigned char)as_string(v)->data[0];
}

SkValue sk_neg(SkValue a) {
    if (a.tag == SK_INT) {
        return v_int((int64_t)(0 - (uint64_t)a.payload));
    }
    if (a.tag == SK_FLOAT) return v_float(-as_float(a));
    sk_fail("unsupported operand for unary -");
    return v_none();
}

SkValue sk_pos(SkValue a) {
    if (a.tag == SK_INT) return a;
    if (a.tag == SK_FLOAT) return a;
    sk_fail("unsupported operand for unary +");
    return v_none();
}

SkValue sk_bitnot(SkValue a) {
    return v_int(~need_int(a, "~"));
}

SkValue sk_not(SkValue a) {
    return v_bool(!sk_truthy(a));
}

SkValue sk_contains(SkValue needle, SkValue haystack) {
    if (haystack.tag == SK_STRING) {
        if (needle.tag != SK_STRING) sk_fail("'in <string>' requires string as left operand");
        SkString *h = as_string(haystack), *n = as_string(needle);
        if (n->len == 0) return v_bool(1);
        if (n->len > h->len) return v_bool(0);
        for (int64_t i = 0; i + n->len <= h->len; i++) {
            if (memcmp(h->data + i, n->data, (size_t)n->len) == 0) return v_bool(1);
        }
        return v_bool(0);
    }
    if (haystack.tag == SK_INT_LIST) {
        if (needle.tag != SK_INT) return v_bool(0);
        SkIntList *l = as_int_list(haystack);
        for (int64_t i = 0; i < l->len; i++) {
            if (l->items[i] == needle.payload) return v_bool(1);
        }
        return v_bool(0);
    }
    if (haystack.tag == SK_LIST || haystack.tag == SK_TUPLE) {
        SkList *l = as_list(haystack);
        for (int64_t i = 0; i < l->len; i++) {
            if (value_equal(l->items[i], needle)) return v_bool(1);
        }
        return v_bool(0);
    }
    if (haystack.tag == SK_DICT || haystack.tag == SK_STRUCT) {
        SkDict *d = as_dict(haystack);
        return v_bool(dict_find(d, needle) >= 0);
    }
    if (haystack.tag == SK_RANGE) {
        if (needle.tag != SK_INT) return v_bool(0);
        SkRange *r = as_range(haystack);
        int64_t x = needle.payload;
        if (r->step > 0) {
            if (x < r->start || x >= r->stop) return v_bool(0);
        } else {
            if (x > r->start || x <= r->stop) return v_bool(0);
        }
        return v_bool((x - r->start) % r->step == 0);
    }
    sk_fail("unsupported operand for in");
    return v_bool(0);
}

/* ------------------------------------------------------------------------- */
/* Stringification                                                           */
/* ------------------------------------------------------------------------- */

typedef struct Buf {
    char *data;
    size_t len;
    size_t cap;
} Buf;

static void buf_init(Buf *b) {
    b->data = NULL;
    b->len = 0;
    b->cap = 0;
}

static void buf_reserve(Buf *b, size_t need) {
    if (need + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap : 32;
    while (cap < need + 1) {
        if (cap > SIZE_MAX / 2) {
            cap = need + 1;
            break;
        }
        cap *= 2;
    }
    b->data = (char *)xrealloc(b->data, cap);
    b->cap = cap;
}

static void buf_putn(Buf *b, const char *s, size_t n) {
    buf_reserve(b, b->len + n);
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
}

static void buf_puts(Buf *b, const char *s) {
    buf_putn(b, s, strlen(s));
}

static void buf_putc(Buf *b, char c) {
    buf_putn(b, &c, 1);
}

static void buf_printf(Buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) {
        va_end(ap2);
        sk_fail("formatting error");
    }
    buf_reserve(b, b->len + (size_t)n);
    vsnprintf(b->data + b->len, b->cap - b->len, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
}

static SkValue buf_value(Buf *b) {
    if (!b->data) return sk_string("", 0);
    SkValue v = sk_string(b->data, (int64_t)b->len);
    xfree(b->data);
    b->data = NULL;
    b->len = 0;
    b->cap = 0;
    return v;
}

static void repr_string(Buf *b, SkString *s) {
    buf_putc(b, '"');
    for (int64_t i = 0; i < s->len; i++) {
        unsigned char c = (unsigned char)s->data[i];
        switch (c) {
        case '\\':
            buf_puts(b, "\\\\");
            break;
        case '"':
            buf_puts(b, "\\\"");
            break;
        case '\n':
            buf_puts(b, "\\n");
            break;
        case '\r':
            buf_puts(b, "\\r");
            break;
        case '\t':
            buf_puts(b, "\\t");
            break;
        default:
            if (c >= 0x20 && c < 0x7f) {
                buf_putc(b, (char)c);
            } else {
                buf_printf(b, "\\x%02x", c);
            }
        }
    }
    buf_putc(b, '"');
}

static void repr_into(Buf *b, SkValue v) {
    switch (v.tag) {
    case SK_NONE:
        buf_puts(b, "None");
        break;
    case SK_UNBOUND:
        buf_puts(b, "<unbound>");
        break;
    case SK_BOOL:
        buf_puts(b, v.payload ? "True" : "False");
        break;
    case SK_INT:
        buf_printf(b, "%" PRId64, v.payload);
        break;
    case SK_FLOAT: {
        double d = as_float(v);
        if (isnan(d)) {
            buf_puts(b, "nan");
        } else if (isinf(d)) {
            buf_puts(b, d < 0 ? "-inf" : "inf");
        } else {
            char tmp[64];
            snprintf(tmp, sizeof(tmp), "%.12g", d);
            if (!strchr(tmp, '.') && !strchr(tmp, 'e') && !strchr(tmp, 'E')) {
                strncat(tmp, ".0", sizeof(tmp) - strlen(tmp) - 1);
            }
            buf_puts(b, tmp);
        }
        break;
    }
    case SK_STRING:
        repr_string(b, as_string(v));
        break;
    case SK_CHAR: {
        struct {
            int64_t len;
            char data[1];
        } tmp = {1, {(char)v.payload}};
        repr_string(b, (SkString *)&tmp);
        break;
    }
    case SK_STRING_ELEMS:
        repr_string(b, as_string(v));
        break;
    case SK_INT_LIST: {
        SkIntList *l = as_int_list(v);
        buf_putc(b, '[');
        for (int64_t i = 0; i < l->len; i++) {
            if (i) buf_puts(b, ", ");
            buf_printf(b, "%" PRId64, l->items[i]);
        }
        buf_putc(b, ']');
        break;
    }
    case SK_LIST: {
        SkList *l = as_list(v);
        buf_putc(b, '[');
        for (int64_t i = 0; i < l->len; i++) {
            if (i) buf_puts(b, ", ");
            repr_into(b, l->items[i]);
        }
        buf_putc(b, ']');
        break;
    }
    case SK_TUPLE: {
        SkList *l = as_list(v);
        buf_putc(b, '(');
        for (int64_t i = 0; i < l->len; i++) {
            if (i) buf_puts(b, ", ");
            repr_into(b, l->items[i]);
        }
        if (l->len == 1) buf_putc(b, ',');
        buf_putc(b, ')');
        break;
    }
    case SK_DICT: {
        SkDict *d = as_dict(v);
        buf_putc(b, '{');
        for (int64_t i = 0; i < d->len; i++) {
            if (i) buf_puts(b, ", ");
            repr_into(b, d->items[i].key);
            buf_puts(b, ": ");
            repr_into(b, d->items[i].value);
        }
        buf_putc(b, '}');
        break;
    }
    case SK_STRUCT: {
        SkDict *d = as_dict(v);
        buf_puts(b, "struct(");
        for (int64_t i = 0; i < d->len; i++) {
            if (i) buf_puts(b, ", ");
            SkString *name = as_string(d->items[i].key);
            buf_putn(b, name->data, (size_t)name->len);
            buf_puts(b, "=");
            repr_into(b, d->items[i].value);
        }
        buf_putc(b, ')');
        break;
    }
    case SK_RANGE: {
        SkRange *r = as_range(v);
        buf_printf(b, "range(%" PRId64 ", %" PRId64 ", %" PRId64 ")", r->start, r->stop, r->step);
        break;
    }
    case SK_FUNCTION:
        buf_puts(b, "<function>");
        break;
    case SK_BUILTIN:
        buf_puts(b, "<built-in function>");
        break;
    case SK_METHOD:
        buf_puts(b, "<method>");
        break;
    default:
        buf_printf(b, "<value:%" PRId64 ">", v.tag);
    }
}

static SkValue value_repr(SkValue v) {
    Buf b;
    buf_init(&b);
    repr_into(&b, v);
    return buf_value(&b);
}

static SkValue value_str(SkValue v) {
    if (v.tag == SK_STRING) return v;
    return value_repr(v);
}

SkValue sk_repr(SkValue v) {
    return value_repr(v);
}

SkValue sk_str(SkValue v) {
    return value_str(v);
}

static char *value_to_cstr(SkValue v) {
    if (v.tag == SK_STRING) {
        SkString *s = as_string(v);
        char *out = (char *)xmalloc((size_t)s->len + 1);
        memcpy(out, s->data, (size_t)s->len);
        out[s->len] = '\0';
        return out;
    }
    SkValue s = value_str(v);
    SkString *str = as_string(s);
    char *out = (char *)xmalloc((size_t)str->len + 1);
    memcpy(out, str->data, (size_t)str->len);
    out[str->len] = '\0';
    return out;
}

int64_t sk_string_eq_cstr(SkValue v, const char *s, int64_t len) {
    if (v.tag != SK_STRING) return 0;
    SkString *x = as_string(v);
    return x->len == len && memcmp(x->data, s, (size_t)len) == 0;
}

/* ------------------------------------------------------------------------- */
/* String formatting                                                         */
/* ------------------------------------------------------------------------- */

static SkValue format_percent(SkValue fmt, SkValue arg) {
    SkString *f = as_string(fmt);
    Buf out;
    buf_init(&out);
    int64_t arg_index = 0;
    SkList *args = (arg.tag == SK_TUPLE) ? as_list(arg) : NULL;

    for (int64_t i = 0; i < f->len; i++) {
        char c = f->data[i];
        if (c != '%') {
            buf_putc(&out, c);
            continue;
        }
        if (i + 1 >= f->len) sk_fail("incomplete format specifier");
        i++;
        if (f->data[i] == '%') {
            buf_putc(&out, '%');
            continue;
        }

        SkValue value;
        if (f->data[i] == '(') {
            int64_t j = i + 1;
            while (j < f->len && f->data[j] != ')') j++;
            if (j >= f->len) sk_fail("unterminated format key");
            SkValue key = sk_string(f->data + i + 1, j - i - 1);
            if (arg.tag != SK_DICT) sk_fail("format key requires a dictionary");
            value = sk_dict_get_impl(arg, key, 1, v_none());
            i = j;
            if (i + 1 >= f->len) sk_fail("incomplete format specifier");
            i++;
        } else {
            if (args) {
                if (arg_index >= args->len) sk_fail("not enough arguments for format string");
                value = args->items[arg_index++];
            } else {
                if (arg_index > 0) sk_fail("not enough arguments for format string");
                value = arg;
                arg_index++;
            }
        }

        char conv = f->data[i];
        char spec[32];
        size_t sl = 0;
        spec[sl++] = '%';
        // Re-read width/precision between the value and conversion. We already
        // advanced to the conversion character, so use a simple direct path.
        switch (conv) {
        case 's':
        case 'r': {
            SkValue s = conv == 'r' ? value_repr(value) : value_str(value);
            SkString *str = as_string(s);
            buf_putn(&out, str->data, (size_t)str->len);
            break;
        }
        case 'd':
        case 'i':
        case 'x':
        case 'X':
        case 'o':
        case 'c': {
            int64_t n;
            if (value.tag == SK_INT) n = value.payload;
            else if (value.tag == SK_BOOL) n = value.payload;
            else if (value.tag == SK_FLOAT) n = (int64_t)as_float(value);
            else sk_fail("integer format requires an integer");
            if (conv == 'c') {
                buf_putc(&out, (char)n);
            } else {
                const char *fmt = conv == 'd' || conv == 'i' ? "lld"
                                : conv == 'x' ? "llx"
                                : conv == 'X' ? "llX"
                                : "llo";
                snprintf(spec + sl, sizeof(spec) - sl, "%s", fmt);
                buf_printf(&out, spec, (long long)n);
            }
            break;
        }
        case 'f':
        case 'F':
        case 'g':
        case 'G':
        case 'e':
        case 'E': {
            double d = value.tag == SK_INT ? (double)value.payload : as_float(value);
            char fmt[2] = {conv, '\0'};
            snprintf(spec + sl, sizeof(spec) - sl, "%s", fmt);
            buf_printf(&out, spec, d);
            break;
        }
        default:
            sk_failf("unsupported format character '%c'", conv);
        }
    }
    if (args && arg_index != args->len) sk_fail("not all arguments converted during string formatting");
    return buf_value(&out);
}

/* ------------------------------------------------------------------------- */
/* Indexing and slicing                                                      */
/* ------------------------------------------------------------------------- */

static int64_t normalize_index(int64_t i, int64_t len) {
    if (i < 0) i += len;
    if (i < 0 || i >= len) sk_fail("index out of range");
    return i;
}

int64_t sk_int_list_get(SkValue list, int64_t index) {
    if (list.tag == SK_INT_LIST) {
        SkIntList *l = as_int_list(list);
        if (index < 0) index += l->len;
        if (index < 0 || index >= l->len) sk_fail("list index out of range");
        return l->items[index];
    }
    if (list.tag != SK_LIST) sk_fail("int list expected");
    SkList *l = as_list(list);
    if (index < 0) index += l->len;
    if (index < 0 || index >= l->len) sk_fail("list index out of range");
    SkValue v = l->items[index];
    if (v.tag != SK_INT) sk_fail("int list element expected");
    return v.payload;
}

SkValue sk_index(SkValue container, SkValue index) {
    if (container.tag == SK_DICT) return sk_dict_get_impl(container, index, 1, v_none());
    if (index.tag != SK_INT) sk_fail("sequence index must be an integer");
    int64_t i = index.payload;
    switch (container.tag) {
    case SK_INT_LIST: {
        SkIntList *l = as_int_list(container);
        return v_int(l->items[normalize_index(i, l->len)]);
    }
    case SK_LIST:
    case SK_TUPLE: {
        SkList *l = as_list(container);
        return l->items[normalize_index(i, l->len)];
    }
    case SK_STRING: {
        SkString *s = as_string(container);
        i = normalize_index(i, s->len);
        return sk_string(s->data + i, 1);
    }
    case SK_STRING_ELEMS: {
        SkString *s = as_string(container);
        i = normalize_index(i, s->len);
        return (SkValue){SK_CHAR, (unsigned char)s->data[i]};
    }
    case SK_RANGE: {
        SkRange *r = as_range(container);
        int64_t n = range_len(r);
        i = normalize_index(i, n);
        return v_int(r->start + i * r->step);
    }
    default:
        sk_fail("value is not indexable");
        return v_none();
    }
}

void sk_set_index(SkValue container, SkValue index, SkValue value) {
    if (container.tag == SK_DICT) {
        sk_dict_set(container, index, value);
        return;
    }
    if (container.tag == SK_INT_LIST) {
        if (index.tag != SK_INT) sk_fail("list index must be an integer");
        if (value.tag != SK_INT) sk_fail("int list element expected");
        SkIntList *l = as_int_list(container);
        l->items[normalize_index(index.payload, l->len)] = value.payload;
        return;
    }
    if (container.tag != SK_LIST) sk_fail("item assignment requires a list or dict");
    if (index.tag != SK_INT) sk_fail("list index must be an integer");
    SkList *l = as_list(container);
    l->items[normalize_index(index.payload, l->len)] = value;
}

static void normalize_slice(int64_t len, int64_t *start, int64_t *stop, int64_t *step,
                            int has_start, int has_stop, int has_step) {
    if (!has_step) *step = 1;
    if (*step == 0) sk_fail("slice step cannot be zero");
    if (*step > 0) {
        if (!has_start) {
            *start = 0;
        } else {
            if (*start < 0) *start += len;
            if (*start < 0) *start = 0;
            if (*start > len) *start = len;
        }
        if (!has_stop) {
            *stop = len;
        } else {
            if (*stop < 0) *stop += len;
            if (*stop < 0) *stop = 0;
            if (*stop > len) *stop = len;
        }
    } else {
        if (!has_start) {
            *start = len - 1;
        } else {
            if (*start < 0) *start += len;
            if (*start < 0) *start = -1;
            if (*start >= len) *start = len - 1;
        }
        if (!has_stop) {
            *stop = -1;
        } else {
            if (*stop < 0) *stop += len;
            if (*stop < 0) *stop = -1;
            if (*stop >= len) *stop = len - 1;
        }
    }
}

SkValue sk_slice(SkValue container, SkValue start_v, SkValue stop_v, SkValue step_v,
                 int64_t has_start, int64_t has_stop, int64_t has_step) {
    if (has_start && start_v.tag != SK_INT) sk_fail("slice start must be an integer");
    if (has_stop && stop_v.tag != SK_INT) sk_fail("slice stop must be an integer");
    if (has_step && step_v.tag != SK_INT) sk_fail("slice step must be an integer");
    int64_t start = start_v.tag == SK_INT ? start_v.payload : 0;
    int64_t stop = stop_v.tag == SK_INT ? stop_v.payload : 0;
    int64_t step = step_v.tag == SK_INT ? step_v.payload : 1;
    int64_t len = sk_iter_len_impl(container);
    normalize_slice(len, &start, &stop, &step, (int)has_start, (int)has_stop, (int)has_step);

    if (container.tag == SK_RANGE) {
        SkRange *r = as_range(container);
        int64_t count = 0;
        for (int64_t i = start; step > 0 ? i < stop : i > stop; i += step) count++;
        int64_t new_start = r->start + start * r->step;
        int64_t new_step = r->step * step;
        int64_t new_stop = new_start + count * new_step;
        return sk_range(new_start, new_stop, new_step);
    }

    if (container.tag == SK_STRING) {
        SkString *s = as_string(container);
        Buf b;
        buf_init(&b);
        for (int64_t i = start; step > 0 ? i < stop : i > stop; i += step) {
            buf_putc(&b, s->data[i]);
        }
        return buf_value(&b);
    }

    if (container.tag == SK_INT_LIST) {
        SkIntList *l = as_int_list(container);
        int64_t count = 0;
        for (int64_t i = start; step > 0 ? i < stop : i > stop; i += step) count++;
        SkValue out = sk_int_list_new_from(count, NULL);
        SkIntList *o = as_int_list(out);
        if (step == 1) {
            if (count) memcpy(o->items, l->items + start, sizeof(int64_t) * (size_t)count);
        } else {
            int64_t j = 0;
            for (int64_t i = start; step > 0 ? i < stop : i > stop; i += step) o->items[j++] = l->items[i];
        }
        return out;
    }
    if (container.tag == SK_LIST || container.tag == SK_TUPLE) {
        SkList *l = as_list(container);
        int64_t count = 0;
        for (int64_t i = start; step > 0 ? i < stop : i > stop; i += step) count++;
        SkValue out = container.tag == SK_LIST ? sk_list_new_from(count, NULL)
                                               : sk_tuple_new(count, NULL);
        SkList *o = as_list(out);
        int64_t j = 0;
        for (int64_t i = start; step > 0 ? i < stop : i > stop; i += step) {
            o->items[j++] = l->items[i];
        }
        return out;
    }
    sk_fail("value is not sliceable");
    return v_none();
}

/* The compiler emits this for slices whose receiver was inferred as a compact
 * integer list.  It keeps the hot path free of boxed slice arguments and the
 * generic tag switch in sk_slice. */
SkValue sk_int_list_slice(SkValue container, int64_t start, int64_t stop, int64_t step,
                          int64_t has_start, int64_t has_stop, int64_t has_step) {
    if (container.tag != SK_INT_LIST) sk_fail("int list slice expected");
    SkIntList *l = as_int_list(container);
    normalize_slice(l->len, &start, &stop, &step, (int)has_start, (int)has_stop,
                    (int)has_step);
    int64_t count = 0;
    for (int64_t i = start; step > 0 ? i < stop : i > stop; i += step) count++;
    SkValue out = sk_int_list_new_from(count, NULL);
    SkIntList *o = as_int_list(out);
    if (step == 1) {
        if (count) memcpy(o->items, l->items + start, sizeof(int64_t) * (size_t)count);
    } else {
        int64_t j = 0;
        for (int64_t i = start; step > 0 ? i < stop : i > stop; i += step) {
            o->items[j++] = l->items[i];
        }
    }
    return out;
}

/* ------------------------------------------------------------------------- */
/* Call argument helpers                                                     */
/* ------------------------------------------------------------------------- */

SkCallArgs *sk_callargs_new(void) {
    SkCallArgs *a = (SkCallArgs *)xmalloc(sizeof(SkCallArgs));
    a->nargs = 0;
    a->cap = 0;
    a->args = NULL;
    a->nkwargs = 0;
    a->kwcap = 0;
    a->kw_names = NULL;
    a->kw_values = NULL;
    return a;
}

static void callargs_reserve_args(SkCallArgs *a, int64_t need) {
    if (need <= a->cap) return;
    int64_t cap = a->cap ? a->cap : 4;
    while (cap < need) cap *= 2;
    a->args = (SkValue *)xrealloc(a->args, sizeof(SkValue) * (size_t)cap);
    a->cap = cap;
}

static void callargs_reserve_kw(SkCallArgs *a, int64_t need) {
    if (need <= a->kwcap) return;
    int64_t cap = a->kwcap ? a->kwcap : 4;
    while (cap < need) cap *= 2;
    a->kw_names = (SkValue *)xrealloc(a->kw_names, sizeof(SkValue) * (size_t)cap);
    a->kw_values = (SkValue *)xrealloc(a->kw_values, sizeof(SkValue) * (size_t)cap);
    a->kwcap = cap;
}

void sk_callargs_arg(SkCallArgs *a, SkValue v) {
    callargs_reserve_args(a, a->nargs + 1);
    a->args[a->nargs++] = v;
}

void sk_callargs_args(SkCallArgs *a, SkValue iterable) {
    int64_t n = sk_iter_len_impl(iterable);
    for (int64_t i = 0; i < n; i++) sk_callargs_arg(a, sk_iter_at_impl(iterable, i));
}

void sk_callargs_kw(SkCallArgs *a, SkValue name, SkValue value) {
    if (name.tag != SK_STRING) sk_fail("keyword name must be a string");
    callargs_reserve_kw(a, a->nkwargs + 1);
    a->kw_names[a->nkwargs] = name;
    a->kw_values[a->nkwargs] = value;
    a->nkwargs++;
}

void sk_callargs_kwargs(SkCallArgs *a, SkValue dict) {
    if (dict.tag != SK_DICT) sk_fail("** argument must be a dictionary");
    SkDict *d = as_dict(dict);
    for (int64_t i = 0; i < d->len; i++) {
        sk_callargs_kw(a, d->items[i].key, d->items[i].value);
    }
}

static SkValue arg_at(SkCallArgs *a, int64_t i) {
    if (i < 0 || i >= a->nargs) sk_fail("missing argument");
    return a->args[i];
}

static SkValue arg_or(SkCallArgs *a, int64_t i, SkValue def) {
    return i < a->nargs ? a->args[i] : def;
}

static SkValue kw_or(SkCallArgs *a, const char *name, SkValue def) {
    for (int64_t i = 0; i < a->nkwargs; i++) {
        SkString *s = as_string(a->kw_names[i]);
        if ((int64_t)strlen(name) == s->len && memcmp(s->data, name, (size_t)s->len) == 0) {
            return a->kw_values[i];
        }
    }
    return def;
}

static int kw_has(SkCallArgs *a, const char *name) {
    for (int64_t i = 0; i < a->nkwargs; i++) {
        SkString *s = as_string(a->kw_names[i]);
        if ((int64_t)strlen(name) == s->len && memcmp(s->data, name, (size_t)s->len) == 0) {
            return 1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Bound methods                                                             */
/* ------------------------------------------------------------------------- */

enum {
    M_APPEND = 1,
    M_EXTEND,
    M_INSERT,
    M_REMOVE,
    M_POP,
    M_CLEAR,
    M_INDEX,
    M_COUNT,
    M_GET,
    M_KEYS,
    M_VALUES,
    M_ITEMS,
    M_UPDATE,
    M_SETDEFAULT,
    M_FORMAT,
    M_JOIN,
    M_SPLIT,
    M_RSPLIT,
    M_STRIP,
    M_LSTRIP,
    M_RSTRIP,
    M_STARTSWITH,
    M_ENDSWITH,
    M_REPLACE,
    M_UPPER,
    M_LOWER,
    M_FIND,
    M_RFIND,
    M_ISDIGIT,
    M_ISALPHA,
    M_ISALNUM,
    M_ISSPACE,
    M_TITLE,
    M_CAPITALIZE,
    M_ELEMS,
    M_REVERSE,
    M_SORT,
    M_COPY,
    M_POPITEM
};

static int string_is(SkValue name, const char *s) {
    if (name.tag != SK_STRING) return 0;
    SkString *x = as_string(name);
    size_t n = strlen(s);
    return x->len == (int64_t)n && memcmp(x->data, s, n) == 0;
}

static int64_t method_id_for(SkValue obj, SkValue name) {
    if (obj.tag == SK_LIST || obj.tag == SK_INT_LIST || obj.tag == SK_TUPLE) {
        if (obj.tag == SK_LIST) {
            if (string_is(name, "append")) return M_APPEND;
            if (string_is(name, "extend")) return M_EXTEND;
            if (string_is(name, "insert")) return M_INSERT;
            if (string_is(name, "remove")) return M_REMOVE;
            if (string_is(name, "pop")) return M_POP;
            if (string_is(name, "clear")) return M_CLEAR;
            if (string_is(name, "reverse")) return M_REVERSE;
            if (string_is(name, "sort")) return M_SORT;
        }
        if (string_is(name, "index")) return M_INDEX;
        if (string_is(name, "count")) return M_COUNT;
    } else if (obj.tag == SK_DICT) {
        if (string_is(name, "get")) return M_GET;
        if (string_is(name, "keys")) return M_KEYS;
        if (string_is(name, "values")) return M_VALUES;
        if (string_is(name, "items")) return M_ITEMS;
        if (string_is(name, "update")) return M_UPDATE;
        if (string_is(name, "pop")) return M_POPITEM;
        if (string_is(name, "setdefault")) return M_SETDEFAULT;
        if (string_is(name, "clear")) return M_CLEAR;
        if (string_is(name, "copy")) return M_COPY;
    } else if (obj.tag == SK_STRING) {
        if (string_is(name, "format")) return M_FORMAT;
        if (string_is(name, "join")) return M_JOIN;
        if (string_is(name, "split")) return M_SPLIT;
        if (string_is(name, "rsplit")) return M_RSPLIT;
        if (string_is(name, "strip")) return M_STRIP;
        if (string_is(name, "lstrip")) return M_LSTRIP;
        if (string_is(name, "rstrip")) return M_RSTRIP;
        if (string_is(name, "startswith")) return M_STARTSWITH;
        if (string_is(name, "endswith")) return M_ENDSWITH;
        if (string_is(name, "replace")) return M_REPLACE;
        if (string_is(name, "upper")) return M_UPPER;
        if (string_is(name, "lower")) return M_LOWER;
        if (string_is(name, "find")) return M_FIND;
        if (string_is(name, "rfind")) return M_RFIND;
        if (string_is(name, "count")) return M_COUNT;
        if (string_is(name, "isdigit")) return M_ISDIGIT;
        if (string_is(name, "isalpha")) return M_ISALPHA;
        if (string_is(name, "isalnum")) return M_ISALNUM;
        if (string_is(name, "isspace")) return M_ISSPACE;
        if (string_is(name, "title")) return M_TITLE;
        if (string_is(name, "capitalize")) return M_CAPITALIZE;
        if (string_is(name, "elems")) return M_ELEMS;
    }
    return 0;
}

SkValue sk_getattr(SkValue obj, SkValue name) {
    if (obj.tag == SK_STRUCT) {
        SkDict *d = as_dict(obj);
        int64_t i = dict_find(d, name);
        if (i < 0) {
            SkValue s = value_str(name);
            SkString *str = as_string(s);
            sk_failf("struct has no field '%.*s'", (int)str->len, str->data);
        }
        return d->items[i].value;
    }
    int64_t id = method_id_for(obj, name);
    if (!id) {
        SkValue s = value_str(name);
        SkString *str = as_string(s);
        sk_failf("object has no attribute '%.*s'", (int)str->len, str->data);
    }
    SkMethod *m = (SkMethod *)xmalloc(sizeof(SkMethod));
    m->recv = obj;
    m->id = id;
    return (SkValue){SK_METHOD, (int64_t)(intptr_t)m};
}

SkValue sk_elems(SkValue v) {
    if (v.tag != SK_STRING) sk_fail("elems() requires a string");
    return (SkValue){SK_STRING_ELEMS, v.payload};
}

void sk_append(SkValue list, SkValue item) {
    sk_list_push(list, item);
}

void sk_setattr(SkValue obj, SkValue name, SkValue value) {
    if (obj.tag == SK_STRUCT) {
        sk_dict_set(obj, name, value);
        return;
    }
    sk_fail("attribute assignment is not supported");
}

static SkValue method_call(SkValue recv, int64_t id, SkCallArgs *a);

/* ------------------------------------------------------------------------- */
/* Built-in functions                                                        */
/* ------------------------------------------------------------------------- */

enum {
    B_PRINT = 1,
    B_LEN,
    B_STR,
    B_REPR,
    B_TYPE,
    B_INT,
    B_FLOAT,
    B_BOOL,
    B_LIST,
    B_TUPLE,
    B_DICT,
    B_RANGE,
    B_ENUMERATE,
    B_ZIP,
    B_REVERSED,
    B_SORTED,
    B_MIN,
    B_MAX,
    B_SUM,
    B_ABS,
    B_ALL,
    B_ANY,
    B_FAIL,
    B_GETATTR,
    B_HASATTR,
    B_DIR,
    B_HASH,
    B_ORD,
    B_CHR,
    B_CALLABLE,
    B_STRUCT
};

static SkValue convert_to_list(SkValue v) {
    int64_t n = sk_iter_len_impl(v);
    SkValue out = sk_list_new_from(n, NULL);
    SkList *l = as_list(out);
    for (int64_t i = 0; i < n; i++) l->items[i] = sk_iter_at_impl(v, i);
    return out;
}

static SkValue convert_to_tuple(SkValue v) {
    int64_t n = sk_iter_len_impl(v);
    SkValue out = sk_tuple_new(n, NULL);
    SkList *l = as_list(out);
    for (int64_t i = 0; i < n; i++) l->items[i] = sk_iter_at_impl(v, i);
    return out;
}

static int64_t parse_int_string(SkString *s, int base) {
    char *tmp = (char *)xmalloc((size_t)s->len + 1);
    memcpy(tmp, s->data, (size_t)s->len);
    tmp[s->len] = '\0';
    char *end = NULL;
    errno = 0;
    long long v = strtoll(tmp, &end, base);
    if (errno || end == tmp) {
        xfree(tmp);
        sk_fail("invalid integer literal");
    }
    while (*end && isspace((unsigned char)*end)) end++;
    if (*end) {
        xfree(tmp);
        sk_fail("invalid integer literal");
    }
    xfree(tmp);
    return (int64_t)v;
}

static SkValue builtin_call(int64_t id, SkCallArgs *a) {
    switch (id) {
    case B_PRINT: {
        SkValue sep = kw_or(a, "sep", sk_string(" ", 1));
        SkValue end = kw_or(a, "end", sk_string("\n", 1));
        char *sep_s = value_to_cstr(sep);
        char *end_s = value_to_cstr(end);
        for (int64_t i = 0; i < a->nargs; i++) {
            if (i) fputs(sep_s, stdout);
            char *s = value_to_cstr(a->args[i]);
            fputs(s, stdout);
            xfree(s);
        }
        fputs(end_s, stdout);
        xfree(sep_s);
        xfree(end_s);
        return v_none();
    }
    case B_LEN:
        return v_int(sk_iter_len_impl(arg_at(a, 0)));
    case B_STR:
        return value_str(arg_at(a, 0));
    case B_REPR:
        return value_repr(arg_at(a, 0));
    case B_TYPE: {
        SkValue v = arg_at(a, 0);
        switch (v.tag) {
        case SK_NONE:
            return sk_string_cstr("NoneType");
        case SK_BOOL:
            return sk_string_cstr("bool");
        case SK_INT:
            return sk_string_cstr("int");
        case SK_FLOAT:
            return sk_string_cstr("float");
        case SK_STRING:
            return sk_string_cstr("string");
        case SK_LIST:
            return sk_string_cstr("list");
        case SK_TUPLE:
            return sk_string_cstr("tuple");
        case SK_DICT:
            return sk_string_cstr("dict");
        case SK_STRUCT:
            return sk_string_cstr("struct");
        case SK_RANGE:
            return sk_string_cstr("range");
        case SK_FUNCTION:
            return sk_string_cstr("function");
        case SK_BUILTIN:
            return sk_string_cstr("builtin_function_or_method");
        case SK_METHOD:
            return sk_string_cstr("builtin_function_or_method");
        default:
            return sk_string_cstr("unknown");
        }
    }
    case B_INT: {
        SkValue v = arg_at(a, 0);
        int64_t base = 10;
        if (a->nargs > 1) base = a->args[1].payload;
        else if (kw_has(a, "base")) base = kw_or(a, "base", v_int(10)).payload;
        if (v.tag == SK_INT) return v;
        if (v.tag == SK_BOOL) return v_int(v.payload);
        if (v.tag == SK_FLOAT) return v_int((int64_t)as_float(v));
        if (v.tag == SK_STRING) return v_int(parse_int_string(as_string(v), (int)base));
        sk_fail("int() argument must be a string, number, or bool");
        return v_none();
    }
    case B_FLOAT: {
        SkValue v = arg_at(a, 0);
        if (v.tag == SK_FLOAT) return v;
        if (v.tag == SK_INT) return v_float((double)v.payload);
        if (v.tag == SK_BOOL) return v_float((double)v.payload);
        if (v.tag == SK_STRING) {
            SkString *s = as_string(v);
            char *tmp = (char *)xmalloc((size_t)s->len + 1);
            memcpy(tmp, s->data, (size_t)s->len);
            tmp[s->len] = '\0';
            char *end = NULL;
            errno = 0;
            double d = strtod(tmp, &end);
            if (errno || end == tmp || (*end && !isspace((unsigned char)*end))) {
                xfree(tmp);
                sk_fail("invalid float literal");
            }
            xfree(tmp);
            return v_float(d);
        }
        sk_fail("float() argument must be a string or number");
        return v_none();
    }
    case B_BOOL:
        return v_bool(sk_truthy(arg_at(a, 0)));
    case B_LIST:
        if (a->nargs == 0) return sk_list_new();
        return convert_to_list(arg_at(a, 0));
    case B_TUPLE:
        if (a->nargs == 0) return sk_tuple_new(0, NULL);
        return convert_to_tuple(arg_at(a, 0));
    case B_DICT: {
        SkValue d = sk_dict_new();
        if (a->nargs >= 1) {
            SkValue src = a->args[0];
            if (src.tag == SK_DICT) {
                SkDict *sd = as_dict(src);
                for (int64_t i = 0; i < sd->len; i++) sk_dict_set(d, sd->items[i].key, sd->items[i].value);
            } else {
                int64_t n = sk_iter_len_impl(src);
                for (int64_t i = 0; i < n; i++) {
                    SkValue pair = sk_iter_at_impl(src, i);
                    if (sk_iter_len_impl(pair) != 2) sk_fail("dict() expects key/value pairs");
                    sk_dict_set(d, sk_iter_at_impl(pair, 0), sk_iter_at_impl(pair, 1));
                }
            }
        }
        for (int64_t i = 0; i < a->nkwargs; i++) {
            sk_dict_set(d, a->kw_names[i], a->kw_values[i]);
        }
        return d;
    }
    case B_RANGE: {
        int64_t start = 0, stop = 0, step = 1;
        if (a->nargs == 1) {
            stop = arg_at(a, 0).payload;
        } else if (a->nargs == 2) {
            start = arg_at(a, 0).payload;
            stop = arg_at(a, 1).payload;
        } else if (a->nargs == 3) {
            start = arg_at(a, 0).payload;
            stop = arg_at(a, 1).payload;
            step = arg_at(a, 2).payload;
        } else {
            sk_fail("range() expects 1 to 3 arguments");
        }
        return sk_range(start, stop, step);
    }
    case B_ENUMERATE: {
        SkValue iterable = arg_at(a, 0);
        int64_t start = a->nargs > 1 ? a->args[1].payload : 0;
        int64_t n = sk_iter_len_impl(iterable);
        SkValue out = sk_list_new_from(n, NULL);
        SkList *l = as_list(out);
        for (int64_t i = 0; i < n; i++) {
            SkValue pair[2] = {v_int(start + i), sk_iter_at_impl(iterable, i)};
            l->items[i] = sk_tuple_new(2, pair);
        }
        return out;
    }
    case B_ZIP: {
        int64_t n = INT64_MAX;
        for (int64_t i = 0; i < a->nargs; i++) {
            int64_t m = sk_iter_len_impl(a->args[i]);
            if (m < n) n = m;
        }
        if (a->nargs == 0) n = 0;
        SkValue out = sk_list_new_from(n, NULL);
        SkList *l = as_list(out);
        for (int64_t i = 0; i < n; i++) {
            SkValue *items = (SkValue *)xmalloc(sizeof(SkValue) * (size_t)a->nargs);
            for (int64_t j = 0; j < a->nargs; j++) items[j] = sk_iter_at_impl(a->args[j], i);
            l->items[i] = sk_tuple_new(a->nargs, items);
            xfree(items);
        }
        return out;
    }
    case B_REVERSED: {
        SkValue src = arg_at(a, 0);
        int64_t n = sk_iter_len_impl(src);
        SkValue out = sk_list_new_from(n, NULL);
        SkList *l = as_list(out);
        for (int64_t i = 0; i < n; i++) l->items[i] = sk_iter_at_impl(src, n - 1 - i);
        return out;
    }
    case B_SORTED: {
        SkValue src = arg_at(a, 0);
        SkValue key = kw_or(a, "key", v_none());
        int reverse = sk_truthy(kw_or(a, "reverse", v_bool(0)));
        int64_t n = sk_iter_len_impl(src);
        SkValue out = sk_list_new_from(n, NULL);
        SkList *l = as_list(out);
        for (int64_t i = 0; i < n; i++) l->items[i] = sk_iter_at_impl(src, i);
        for (int64_t i = 1; i < n; i++) {
            SkValue item = l->items[i];
            SkValue item_key = item;
            if (key.tag != SK_NONE) {
                SkCallArgs ka = {1, 1, &item, 0, 0, NULL, NULL};
                item_key = sk_call(key, &ka);
            }
            int64_t j = i - 1;
            while (j >= 0) {
                SkValue other_key = l->items[j];
                if (key.tag != SK_NONE) {
                    SkCallArgs ka = {1, 1, &l->items[j], 0, 0, NULL, NULL};
                    other_key = sk_call(key, &ka);
                }
                int c = value_compare(other_key, item_key);
                if ((!reverse && c <= 0) || (reverse && c >= 0)) break;
                l->items[j + 1] = l->items[j];
                j--;
            }
            l->items[j + 1] = item;
        }
        return out;
    }
    case B_MIN:
    case B_MAX: {
        SkValue iterable;
        int64_t n;
        if (a->nargs == 1) {
            iterable = a->args[0];
            n = sk_iter_len_impl(iterable);
            if (n == 0) sk_fail("min/max of empty sequence");
        } else {
            n = a->nargs;
            iterable = v_none();
        }
        SkValue best = a->nargs == 1 ? sk_iter_at_impl(iterable, 0) : a->args[0];
        SkValue key = kw_or(a, "key", v_none());
        SkValue best_key = best;
        if (key.tag != SK_NONE) {
            SkCallArgs ka = {1, 1, &best, 0, 0, NULL, NULL};
            best_key = sk_call(key, &ka);
        }
        for (int64_t i = 1; i < n; i++) {
            SkValue item = a->nargs == 1 ? sk_iter_at_impl(iterable, i) : a->args[i];
            SkValue item_key = item;
            if (key.tag != SK_NONE) {
                SkCallArgs ka = {1, 1, &item, 0, 0, NULL, NULL};
                item_key = sk_call(key, &ka);
            }
            int c = value_compare(item_key, best_key);
            if ((id == B_MIN && c < 0) || (id == B_MAX && c > 0)) {
                best = item;
                best_key = item_key;
            }
        }
        return best;
    }
    case B_SUM: {
        SkValue sum = a->nargs > 1 ? a->args[1] : v_int(0);
        SkValue src = arg_at(a, 0);
        int64_t n = sk_iter_len_impl(src);
        for (int64_t i = 0; i < n; i++) sum = sk_add(sum, sk_iter_at_impl(src, i));
        return sum;
    }
    case B_ABS: {
        SkValue v = arg_at(a, 0);
        if (v.tag == SK_INT) return v.payload < 0 ? sk_neg(v) : v;
        if (v.tag == SK_FLOAT) return v_float(fabs(as_float(v)));
        sk_fail("abs() expects a number");
        return v_none();
    }
    case B_ALL: {
        SkValue src = arg_at(a, 0);
        int64_t n = sk_iter_len_impl(src);
        for (int64_t i = 0; i < n; i++) {
            if (!sk_truthy(sk_iter_at_impl(src, i))) return v_bool(0);
        }
        return v_bool(1);
    }
    case B_ANY: {
        SkValue src = arg_at(a, 0);
        int64_t n = sk_iter_len_impl(src);
        for (int64_t i = 0; i < n; i++) {
            if (sk_truthy(sk_iter_at_impl(src, i))) return v_bool(1);
        }
        return v_bool(0);
    }
    case B_FAIL: {
        Buf b;
        buf_init(&b);
        for (int64_t i = 0; i < a->nargs; i++) {
            if (i) buf_puts(&b, " ");
            SkValue s = value_str(a->args[i]);
            SkString *str = as_string(s);
            buf_putn(&b, str->data, (size_t)str->len);
        }
        SkValue msg = buf_value(&b);
        SkString *s = as_string(msg);
        sk_failf("%.*s", (int)s->len, s->data);
        return v_none();
    }
    case B_GETATTR: {
        SkValue obj = arg_at(a, 0);
        SkValue name = arg_at(a, 1);
        if (obj.tag == SK_STRUCT) {
            SkDict *d = as_dict(obj);
            int64_t i = dict_find(d, name);
            if (i >= 0) return d->items[i].value;
            if (a->nargs > 2) return a->args[2];
            SkValue s = value_str(name);
            SkString *str = as_string(s);
            sk_failf("struct has no field '%.*s'", (int)str->len, str->data);
        }
        int64_t mid = method_id_for(obj, name);
        if (!mid) {
            if (a->nargs > 2) return a->args[2];
            SkValue s = value_str(name);
            SkString *str = as_string(s);
            sk_failf("object has no attribute '%.*s'", (int)str->len, str->data);
        }
        return sk_getattr(obj, name);
    }
    case B_HASATTR: {
        SkValue obj = arg_at(a, 0);
        SkValue name = arg_at(a, 1);
        if (obj.tag == SK_STRUCT) {
            SkDict *d = as_dict(obj);
            if (dict_find(d, name) >= 0) return v_bool(1);
        }
        return v_bool(method_id_for(obj, name) != 0);
    }
    case B_DIR: {
        SkValue obj = arg_at(a, 0);
        const char *names[32];
        int n = 0;
        if (obj.tag == SK_LIST || obj.tag == SK_INT_LIST) {
            names[n++] = "append"; names[n++] = "extend"; names[n++] = "insert";
            names[n++] = "remove"; names[n++] = "pop"; names[n++] = "clear";
            names[n++] = "index"; names[n++] = "count"; names[n++] = "reverse"; names[n++] = "sort";
        } else if (obj.tag == SK_DICT) {
            names[n++] = "get"; names[n++] = "keys"; names[n++] = "values"; names[n++] = "items";
            names[n++] = "update"; names[n++] = "pop"; names[n++] = "setdefault";
            names[n++] = "clear"; names[n++] = "copy";
        } else if (obj.tag == SK_STRUCT) {
            SkDict *d = as_dict(obj);
            SkValue out = sk_list_new_from(d->len, NULL);
            SkList *l = as_list(out);
            for (int64_t i = 0; i < d->len; i++) l->items[i] = d->items[i].key;
            return out;
        } else if (obj.tag == SK_STRING) {
            names[n++] = "format"; names[n++] = "join"; names[n++] = "split"; names[n++] = "rsplit";
            names[n++] = "strip"; names[n++] = "lstrip"; names[n++] = "rstrip";
            names[n++] = "startswith"; names[n++] = "endswith"; names[n++] = "replace";
            names[n++] = "upper"; names[n++] = "lower"; names[n++] = "find"; names[n++] = "rfind";
            names[n++] = "count"; names[n++] = "isdigit"; names[n++] = "isalpha";
            names[n++] = "isalnum"; names[n++] = "isspace"; names[n++] = "title";
            names[n++] = "capitalize"; names[n++] = "elems";
        }
        SkValue out = sk_list_new_from(n, NULL);
        SkList *l = as_list(out);
        for (int i = 0; i < n; i++) l->items[i] = sk_string_cstr(names[i]);
        return out;
    }
    case B_HASH: {
        SkValue v = arg_at(a, 0);
        if (!is_hashable(v)) sk_fail("unhashable value passed to hash()");
        switch (v.tag) {
        case SK_NONE:
            return v_int(0);
        case SK_INT:
            return v;
        case SK_BOOL:
        case SK_CHAR:
            return v_int(v.payload);
        case SK_FLOAT:
            return v_int(v.payload);
        case SK_STRING: {
            SkString *s = as_string(v);
            uint64_t h = 1469598103934665603ULL;
            for (int64_t i = 0; i < s->len; i++) {
                h ^= (unsigned char)s->data[i];
                h *= 1099511628211ULL;
            }
            return v_int((int64_t)h);
        }
        case SK_TUPLE: {
            SkList *l = as_list(v);
            uint64_t h = 1469598103934665603ULL;
            for (int64_t i = 0; i < l->len; i++) {
                SkValue hv = builtin_call(B_HASH, &(SkCallArgs){1, 1, &l->items[i], 0, 0, NULL, NULL});
                h ^= (uint64_t)hv.payload;
                h *= 1099511628211ULL;
            }
            return v_int((int64_t)h);
        }
        case SK_RANGE: {
            SkRange *r = as_range(v);
            uint64_t h = (uint64_t)r->start;
            h = h * 1099511628211ULL ^ (uint64_t)r->stop;
            h = h * 1099511628211ULL ^ (uint64_t)r->step;
            return v_int((int64_t)h);
        }
        default:
            return v_int(v.payload);
        }
    }
    case B_ORD: {
        SkValue v = arg_at(a, 0);
        if (v.tag != SK_STRING || as_string(v)->len != 1) sk_fail("ord() expects a one-character string");
        return v_int((unsigned char)as_string(v)->data[0]);
    }
    case B_CHR: {
        int64_t n = arg_at(a, 0).payload;
        if (n < 0 || n > 255) sk_fail("chr() argument out of range");
        char c = (char)n;
        return sk_string(&c, 1);
    }
    case B_CALLABLE: {
        SkValue v = arg_at(a, 0);
        return v_bool(v.tag == SK_FUNCTION || v.tag == SK_BUILTIN || v.tag == SK_METHOD);
    }
    case B_STRUCT: {
        if (a->nargs != 0) sk_fail("struct() accepts only keyword arguments");
        SkValue out = sk_struct_new();
        for (int64_t i = 0; i < a->nkwargs; i++) {
            sk_dict_set(out, a->kw_names[i], a->kw_values[i]);
        }
        return out;
    }
    default:
        sk_fail("unknown builtin");
        return v_none();
    }
}

/* ------------------------------------------------------------------------- */
/* String method helpers                                                     */
/* ------------------------------------------------------------------------- */

static SkValue string_from_buf(Buf *b) {
    return buf_value(b);
}

static int64_t find_substring(SkString *hay, SkString *needle, int64_t start, int64_t end, int reverse) {
    if (start < 0) start += hay->len;
    if (end < 0) end += hay->len;
    if (start < 0) start = 0;
    if (end > hay->len) end = hay->len;
    if (end < start) return -1;
    if (needle->len == 0) return reverse ? end : start;
    if (needle->len > end - start) return -1;
    if (!reverse) {
        for (int64_t i = start; i + needle->len <= end; i++) {
            if (memcmp(hay->data + i, needle->data, (size_t)needle->len) == 0) return i;
        }
    } else {
        for (int64_t i = end - needle->len; i >= start; i--) {
            if (memcmp(hay->data + i, needle->data, (size_t)needle->len) == 0) return i;
        }
    }
    return -1;
}

static void strip_chars(SkString *s, SkValue chars, int left, int right, Buf *out) {
    unsigned char mask[256] = {0};
    if (chars.tag == SK_NONE) {
        mask[' '] = mask['\t'] = mask['\n'] = mask['\r'] = mask['\v'] = mask['\f'] = 1;
    } else if (chars.tag == SK_STRING) {
        SkString *c = as_string(chars);
        for (int64_t i = 0; i < c->len; i++) mask[(unsigned char)c->data[i]] = 1;
    } else {
        sk_fail("strip argument must be a string");
    }
    int64_t a = 0, b = s->len;
    if (left) {
        while (a < b && mask[(unsigned char)s->data[a]]) a++;
    }
    if (right) {
        while (b > a && mask[(unsigned char)s->data[b - 1]]) b--;
    }
    buf_putn(out, s->data + a, (size_t)(b - a));
}

static SkValue string_format_method(SkValue recv, SkCallArgs *a) {
    SkString *f = as_string(recv);
    Buf out;
    buf_init(&out);
    int64_t auto_index = 0;
    for (int64_t i = 0; i < f->len; i++) {
        char c = f->data[i];
        if (c == '{') {
            if (i + 1 < f->len && f->data[i + 1] == '{') {
                buf_putc(&out, '{');
                i++;
                continue;
            }
            int64_t j = i + 1;
            int depth = 1;
            while (j < f->len && depth) {
                if (f->data[j] == '{') depth++;
                else if (f->data[j] == '}') depth--;
                if (depth) j++;
            }
            if (j >= f->len) sk_fail("unterminated format field");
            int64_t field_start = i + 1;
            int64_t field_end = j;
            int conv = 0;
            int64_t colon = -1;
            int64_t bang = -1;
            for (int64_t k = field_start; k < field_end; k++) {
                if (f->data[k] == ':' && colon < 0) colon = k;
                if (f->data[k] == '!' && bang < 0) bang = k;
            }
            int64_t name_end = field_end;
            if (colon >= 0 && colon < name_end) name_end = colon;
            if (bang >= 0 && bang < name_end) name_end = bang;
            if (bang >= 0 && bang + 1 < field_end) conv = f->data[bang + 1];

            SkValue value;
            if (name_end == field_start) {
                value = arg_at(a, auto_index++);
            } else if (isdigit((unsigned char)f->data[field_start])) {
                int64_t idx = 0;
                for (int64_t k = field_start; k < name_end; k++) {
                    if (!isdigit((unsigned char)f->data[k])) sk_fail("invalid format field");
                    idx = idx * 10 + (f->data[k] - '0');
                }
                value = arg_at(a, idx);
            } else {
                SkValue key = sk_string(f->data + field_start, name_end - field_start);
                value = kw_or(a, value_to_cstr(key), v_none());
            }
            SkValue s = conv == 'r' ? value_repr(value) : value_str(value);
            SkString *str = as_string(s);
            buf_putn(&out, str->data, (size_t)str->len);
            i = j;
        } else if (c == '}') {
            if (i + 1 < f->len && f->data[i + 1] == '}') {
                buf_putc(&out, '}');
                i++;
            } else {
                sk_fail("single '}' in format string");
            }
        } else {
            buf_putc(&out, c);
        }
    }
    return string_from_buf(&out);
}

static SkValue string_split_method(SkValue recv, SkCallArgs *a, int right) {
    SkString *s = as_string(recv);
    SkValue sep = arg_or(a, 0, v_none());
    int64_t maxsplit = a->nargs > 1 ? a->args[1].payload : -1;
    SkValue out = sk_list_new();
    if (sep.tag == SK_NONE) {
        int64_t i = 0;
        int64_t splits = 0;
        while (i < s->len) {
            while (i < s->len && isspace((unsigned char)s->data[i])) i++;
            if (i >= s->len) break;
            if (maxsplit >= 0 && splits >= maxsplit) {
                sk_list_push(out, sk_string(s->data + i, s->len - i));
                break;
            }
            int64_t start = i;
            while (i < s->len && !isspace((unsigned char)s->data[i])) i++;
            sk_list_push(out, sk_string(s->data + start, i - start));
            splits++;
        }
        return out;
    }
    if (sep.tag != SK_STRING) sk_fail("split separator must be a string");
    SkString *sp = as_string(sep);
    if (sp->len == 0) sk_fail("empty separator");
    int64_t start = 0;
    int64_t splits = 0;
    for (int64_t i = 0; i + sp->len <= s->len;) {
        if (maxsplit >= 0 && splits >= maxsplit) break;
        if (memcmp(s->data + i, sp->data, (size_t)sp->len) == 0) {
            sk_list_push(out, sk_string(s->data + start, i - start));
            i += sp->len;
            start = i;
            splits++;
        } else {
            i++;
        }
    }
    sk_list_push(out, sk_string(s->data + start, s->len - start));
    if (right) {
        SkList *l = as_list(out);
        for (int64_t i = 0; i < l->len / 2; i++) {
            SkValue t = l->items[i];
            l->items[i] = l->items[l->len - 1 - i];
            l->items[l->len - 1 - i] = t;
        }
    }
    return out;
}

static SkValue string_replace_method(SkValue recv, SkCallArgs *a) {
    SkString *s = as_string(recv);
    SkString *old = as_string(arg_at(a, 0));
    SkString *rep = as_string(arg_at(a, 1));
    int64_t count = a->nargs > 2 ? a->args[2].payload : -1;
    Buf out;
    buf_init(&out);
    int64_t i = 0, done = 0;
    while (i < s->len) {
        if ((count < 0 || done < count) && old->len > 0 && i + old->len <= s->len &&
            memcmp(s->data + i, old->data, (size_t)old->len) == 0) {
            buf_putn(&out, rep->data, (size_t)rep->len);
            i += old->len;
            done++;
        } else {
            buf_putc(&out, s->data[i]);
            i++;
        }
    }
    return string_from_buf(&out);
}

static SkValue string_case_method(SkValue recv, int mode) {
    SkString *s = as_string(recv);
    Buf out;
    buf_init(&out);
    int word_start = 1;
    for (int64_t i = 0; i < s->len; i++) {
        unsigned char c = (unsigned char)s->data[i];
        if (mode == 0) {
            buf_putc(&out, (char)tolower(c));
        } else if (mode == 1) {
            buf_putc(&out, (char)toupper(c));
        } else if (mode == 2) {
            if (isalnum(c)) {
                buf_putc(&out, (char)(word_start ? toupper(c) : tolower(c)));
                word_start = 0;
            } else {
                buf_putc(&out, (char)c);
                word_start = 1;
            }
        } else {
            if (i == 0) buf_putc(&out, (char)toupper(c));
            else buf_putc(&out, (char)tolower(c));
        }
    }
    return string_from_buf(&out);
}

static int string_predicate(SkValue recv, int kind) {
    SkString *s = as_string(recv);
    if (s->len == 0) return 0;
    for (int64_t i = 0; i < s->len; i++) {
        unsigned char c = (unsigned char)s->data[i];
        int ok = kind == 0 ? isdigit(c) : kind == 1 ? isalpha(c) : kind == 2 ? isalnum(c) : isspace(c);
        if (!ok) return 0;
    }
    return 1;
}

static SkValue method_call(SkValue recv, int64_t id, SkCallArgs *a) {
    if (recv.tag == SK_INT_LIST) {
        SkIntList *il = as_int_list(recv);
        switch (id) {
        case M_APPEND:
            if (arg_at(a, 0).tag != SK_INT) sk_fail("int list element expected");
            sk_int_list_push(recv, arg_at(a, 0).payload);
            return v_none();
        case M_EXTEND: {
            SkValue src = arg_at(a, 0);
            int64_t n = sk_iter_len_impl(src);
            for (int64_t i = 0; i < n; i++) {
                SkValue item = sk_iter_at_impl(src, i);
                if (item.tag != SK_INT) sk_fail("int list element expected");
                sk_int_list_push(recv, item.payload);
            }
            return v_none();
        }
        case M_POP: {
            int64_t i = a->nargs ? a->args[0].payload : il->len - 1;
            if (i < 0) i += il->len;
            if (i < 0 || i >= il->len) sk_fail("pop index out of range");
            int64_t v = il->items[i];
            for (int64_t j = i + 1; j < il->len; j++) il->items[j - 1] = il->items[j];
            il->len--;
            return v_int(v);
        }
        case M_CLEAR:
            il->len = 0;
            return v_none();
        case M_INDEX: {
            if (arg_at(a, 0).tag != SK_INT) sk_fail("value not found in list");
            int64_t v = arg_at(a, 0).payload;
            int64_t start = a->nargs > 1 ? a->args[1].payload : 0;
            int64_t end = a->nargs > 2 ? a->args[2].payload : il->len;
            if (start < 0) start += il->len;
            if (end < 0) end += il->len;
            if (start < 0) start = 0;
            if (end > il->len) end = il->len;
            for (int64_t i = start; i < end; i++) if (il->items[i] == v) return v_int(i);
            sk_fail("value not found in list");
            return v_none();
        }
        case M_COUNT: {
            if (arg_at(a, 0).tag != SK_INT) return v_int(0);
            int64_t v = arg_at(a, 0).payload, n = 0;
            for (int64_t i = 0; i < il->len; i++) if (il->items[i] == v) n++;
            return v_int(n);
        }
        case M_REVERSE:
            for (int64_t i = 0; i < il->len / 2; i++) {
                int64_t t = il->items[i];
                il->items[i] = il->items[il->len - 1 - i];
                il->items[il->len - 1 - i] = t;
            }
            return v_none();
        default:
            break;
        }
        // Fall back to the generic implementation for operations that are not
        // hot in the hash workloads (sort, insert, remove).
        SkValue generic = sk_list_new_from(il->len, NULL);
        SkList *g = as_list(generic);
        for (int64_t i = 0; i < il->len; i++) g->items[i] = v_int(il->items[i]);
        SkValue result = method_call(generic, id, a);
        il->len = g->len;
        int_list_reserve(il, g->len);
        for (int64_t i = 0; i < g->len; i++) il->items[i] = g->items[i].payload;
        return result;
    }
    if (recv.tag == SK_LIST || recv.tag == SK_TUPLE) {
        SkList *l = as_list(recv);
        switch (id) {
        case M_APPEND:
            sk_list_push(recv, arg_at(a, 0));
            return v_none();
        case M_EXTEND: {
            SkValue src = arg_at(a, 0);
            int64_t n = sk_iter_len_impl(src);
            for (int64_t i = 0; i < n; i++) sk_list_push(recv, sk_iter_at_impl(src, i));
            return v_none();
        }
        case M_INSERT: {
            int64_t i = arg_at(a, 0).payload;
            if (i < 0) i += l->len;
            if (i < 0) i = 0;
            if (i > l->len) i = l->len;
            list_reserve(l, l->len + 1);
            for (int64_t j = l->len; j > i; j--) l->items[j] = l->items[j - 1];
            l->items[i] = arg_at(a, 1);
            l->len++;
            return v_none();
        }
        case M_REMOVE: {
            SkValue v = arg_at(a, 0);
            for (int64_t i = 0; i < l->len; i++) {
                if (value_equal(l->items[i], v)) {
                    for (int64_t j = i + 1; j < l->len; j++) l->items[j - 1] = l->items[j];
                    l->len--;
                    return v_none();
                }
            }
            sk_fail("value not found in list");
            return v_none();
        }
        case M_POP: {
            int64_t i = a->nargs ? a->args[0].payload : l->len - 1;
            if (i < 0) i += l->len;
            if (i < 0 || i >= l->len) sk_fail("pop index out of range");
            SkValue v = l->items[i];
            for (int64_t j = i + 1; j < l->len; j++) l->items[j - 1] = l->items[j];
            l->len--;
            return v;
        }
        case M_CLEAR:
            l->len = 0;
            return v_none();
        case M_INDEX: {
            SkValue v = arg_at(a, 0);
            int64_t start = a->nargs > 1 ? a->args[1].payload : 0;
            int64_t end = a->nargs > 2 ? a->args[2].payload : l->len;
            if (start < 0) start += l->len;
            if (end < 0) end += l->len;
            if (start < 0) start = 0;
            if (end > l->len) end = l->len;
            for (int64_t i = start; i < end; i++) {
                if (value_equal(l->items[i], v)) return v_int(i);
            }
            sk_fail("value not found in list");
            return v_none();
        }
        case M_COUNT: {
            SkValue v = arg_at(a, 0);
            int64_t n = 0;
            for (int64_t i = 0; i < l->len; i++) if (value_equal(l->items[i], v)) n++;
            return v_int(n);
        }
        case M_REVERSE:
            for (int64_t i = 0; i < l->len / 2; i++) {
                SkValue t = l->items[i];
                l->items[i] = l->items[l->len - 1 - i];
                l->items[l->len - 1 - i] = t;
            }
            return v_none();
        case M_SORT: {
            SkValue key = kw_or(a, "key", v_none());
            int reverse = sk_truthy(kw_or(a, "reverse", v_bool(0)));
            for (int64_t i = 1; i < l->len; i++) {
                SkValue item = l->items[i];
                SkValue item_key = item;
                if (key.tag != SK_NONE) {
                    SkCallArgs ka = {1, 1, &item, 0, 0, NULL, NULL};
                    item_key = sk_call(key, &ka);
                }
                int64_t j = i - 1;
                while (j >= 0) {
                    SkValue other_key = l->items[j];
                    if (key.tag != SK_NONE) {
                        SkCallArgs ka = {1, 1, &l->items[j], 0, 0, NULL, NULL};
                        other_key = sk_call(key, &ka);
                    }
                    int c = value_compare(other_key, item_key);
                    if ((!reverse && c <= 0) || (reverse && c >= 0)) break;
                    l->items[j + 1] = l->items[j];
                    j--;
                }
                l->items[j + 1] = item;
            }
            return v_none();
        }
        default:
            break;
        }
    } else if (recv.tag == SK_DICT) {
        SkDict *d = as_dict(recv);
        switch (id) {
        case M_GET:
            return sk_dict_get_impl(recv, arg_at(a, 0), 0, arg_or(a, 1, v_none()));
        case M_KEYS: {
            SkValue out = sk_list_new_from(d->len, NULL);
            SkList *l = as_list(out);
            for (int64_t i = 0; i < d->len; i++) l->items[i] = d->items[i].key;
            return out;
        }
        case M_VALUES: {
            SkValue out = sk_list_new_from(d->len, NULL);
            SkList *l = as_list(out);
            for (int64_t i = 0; i < d->len; i++) l->items[i] = d->items[i].value;
            return out;
        }
        case M_ITEMS: {
            SkValue out = sk_list_new_from(d->len, NULL);
            SkList *l = as_list(out);
            for (int64_t i = 0; i < d->len; i++) {
                SkValue pair[2] = {d->items[i].key, d->items[i].value};
                l->items[i] = sk_tuple_new(2, pair);
            }
            return out;
        }
        case M_UPDATE: {
            SkValue src = arg_at(a, 0);
            if (src.tag == SK_DICT) {
                SkDict *sd = as_dict(src);
                for (int64_t i = 0; i < sd->len; i++) sk_dict_set(recv, sd->items[i].key, sd->items[i].value);
            } else {
                int64_t n = sk_iter_len_impl(src);
                for (int64_t i = 0; i < n; i++) {
                    SkValue pair = sk_iter_at_impl(src, i);
                    if (sk_iter_len_impl(pair) != 2) sk_fail("update expects key/value pairs");
                    sk_dict_set(recv, sk_iter_at_impl(pair, 0), sk_iter_at_impl(pair, 1));
                }
            }
            return v_none();
        }
        case M_POPITEM: {
            SkValue key = arg_at(a, 0);
            int64_t i = dict_find(d, key);
            if (i < 0) {
                if (a->nargs > 1) return a->args[1];
                sk_fail("key not found in dictionary");
            }
            SkValue v = d->items[i].value;
            dict_remove_at(d, i);
            return v;
        }
        case M_SETDEFAULT: {
            SkValue key = arg_at(a, 0);
            int64_t i = dict_find(d, key);
            if (i >= 0) return d->items[i].value;
            SkValue def = arg_or(a, 1, v_none());
            sk_dict_set(recv, key, def);
            return def;
        }
        case M_CLEAR:
            d->len = 0;
            return v_none();
        case M_COPY: {
            SkValue out = sk_dict_new();
            for (int64_t i = 0; i < d->len; i++) sk_dict_set(out, d->items[i].key, d->items[i].value);
            return out;
        }
        default:
            break;
        }
    } else if (recv.tag == SK_STRING) {
        SkString *s = as_string(recv);
        switch (id) {
        case M_FORMAT:
            return string_format_method(recv, a);
        case M_JOIN: {
            SkValue src = arg_at(a, 0);
            int64_t n = sk_iter_len_impl(src);
            Buf out;
            buf_init(&out);
            for (int64_t i = 0; i < n; i++) {
                if (i) buf_putn(&out, s->data, (size_t)s->len);
                SkValue item = sk_iter_at_impl(src, i);
                if (item.tag != SK_STRING) sk_fail("join elements must be strings");
                SkString *is = as_string(item);
                buf_putn(&out, is->data, (size_t)is->len);
            }
            return string_from_buf(&out);
        }
        case M_SPLIT:
            return string_split_method(recv, a, 0);
        case M_RSPLIT:
            return string_split_method(recv, a, 1);
        case M_STRIP:
        case M_LSTRIP:
        case M_RSTRIP: {
            Buf out;
            buf_init(&out);
            strip_chars(s, arg_or(a, 0, v_none()), id != M_RSTRIP, id != M_LSTRIP, &out);
            return string_from_buf(&out);
        }
        case M_STARTSWITH:
        case M_ENDSWITH: {
            SkValue prefix = arg_at(a, 0);
            if (prefix.tag != SK_STRING) sk_fail("prefix must be a string");
            SkString *p = as_string(prefix);
            int64_t start = a->nargs > 1 ? a->args[1].payload : 0;
            int64_t end = a->nargs > 2 ? a->args[2].payload : s->len;
            if (start < 0) start += s->len;
            if (end < 0) end += s->len;
            if (start < 0) start = 0;
            if (end > s->len) end = s->len;
            if (p->len > end - start) return v_bool(0);
            int64_t at = id == M_STARTSWITH ? start : end - p->len;
            return v_bool(memcmp(s->data + at, p->data, (size_t)p->len) == 0);
        }
        case M_REPLACE:
            return string_replace_method(recv, a);
        case M_UPPER:
            return string_case_method(recv, 1);
        case M_LOWER:
            return string_case_method(recv, 0);
        case M_TITLE:
            return string_case_method(recv, 2);
        case M_CAPITALIZE:
            return string_case_method(recv, 3);
        case M_FIND:
        case M_RFIND: {
            SkValue sub = arg_at(a, 0);
            if (sub.tag != SK_STRING) sk_fail("find argument must be a string");
            int64_t start = a->nargs > 1 ? a->args[1].payload : 0;
            int64_t end = a->nargs > 2 ? a->args[2].payload : s->len;
            return v_int(find_substring(s, as_string(sub), start, end, id == M_RFIND));
        }
        case M_COUNT: {
            SkValue sub = arg_at(a, 0);
            if (sub.tag != SK_STRING) sk_fail("count argument must be a string");
            SkString *n = as_string(sub);
            if (n->len == 0) return v_int(s->len + 1);
            int64_t count = 0;
            for (int64_t i = 0; i + n->len <= s->len; i++) {
                if (memcmp(s->data + i, n->data, (size_t)n->len) == 0) {
                    count++;
                    i += n->len - 1;
                }
            }
            return v_int(count);
        }
        case M_ISDIGIT:
            return v_bool(string_predicate(recv, 0));
        case M_ISALPHA:
            return v_bool(string_predicate(recv, 1));
        case M_ISALNUM:
            return v_bool(string_predicate(recv, 2));
        case M_ISSPACE:
            return v_bool(string_predicate(recv, 3));
        case M_ELEMS:
            // A lazy view: iteration yields SK_CHAR values without allocating
            // one string object per byte.  This is especially important for
            // `to_bytes`, which would otherwise allocate a string per input
            // byte before calling ord().
            return (SkValue){SK_STRING_ELEMS, recv.payload};
        default:
            break;
        }
    }
    sk_fail("unknown method");
    return v_none();
}

/* ------------------------------------------------------------------------- */
/* Calls                                                                     */
/* ------------------------------------------------------------------------- */

SkValue sk_call(SkValue callee, SkCallArgs *a) {
    if (callee.tag == SK_FUNCTION) {
        SkClosure *cl = as_closure(callee);
        return cl->code(cl, a->nargs, a->args, a->nkwargs, a->kw_names, a->kw_values);
    }
    if (callee.tag == SK_BUILTIN) return builtin_call(callee.payload, a);
    if (callee.tag == SK_METHOD) {
        SkMethod *m = as_method(callee);
        return method_call(m->recv, m->id, a);
    }
    sk_fail("value is not callable");
    return v_none();
}
