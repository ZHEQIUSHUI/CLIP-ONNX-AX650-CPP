#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <variant>
#include <functional>
#include <iomanip>
#include <ostream>
#include <tuple>
#include <sstream>

namespace tableprinter
{

struct name
{
    explicit name( ::std::string_view val ) : value { val }
    {}

    ::std::string value;

    operator ::std::string() const noexcept { return value; }
};

struct width
{
    explicit width( int val ) : value { val }
    {}

    int value;

    operator int() const noexcept { return value; }
};

struct fill
{
    explicit fill( char val ) : value { val }
    {}

    char value;

    operator char() const noexcept { return value; }
};

struct precision
{
    explicit precision( int val ) : value { val } 
    {}

    int value;

    operator int() const noexcept { return value; }
};

struct default_precision
{};

struct fixed
{};

struct unfixed
{};

struct left
{};

struct right
{};

struct hex
{};

struct decimal
{};

struct octal
{};

using option = ::std::variant<name ,
                              width ,
                              precision ,
                              fill ,
                              default_precision ,
                              fixed ,
                              unfixed ,
                              left ,
                              right ,
                              hex ,
                              decimal ,
                              octal>;

namespace detail
{

template<typename... Ts>
struct overloaded : Ts...
{
    using Ts::operator ()...;
};

template<typename... Ts> overloaded(Ts...) -> overloaded<Ts...>;

template<typename T>
struct is_option :  
       ::std::bool_constant<::std::is_same_v<T , name>              ||
                            ::std::is_same_v<T , width>             ||
                            ::std::is_same_v<T , precision>         ||
                            ::std::is_same_v<T , default_precision> ||
                            ::std::is_same_v<T , fill>              ||
                            ::std::is_same_v<T , fixed>             ||
                            ::std::is_same_v<T , unfixed>           ||
                            ::std::is_same_v<T , left>              ||
                            ::std::is_same_v<T , right>             ||
                            ::std::is_same_v<T , hex>               ||
                            ::std::is_same_v<T , decimal>           ||
                            ::std::is_same_v<T , octal>>
{};

template<typename T>
static constexpr bool is_option_v = is_option<T>::value;

template<typename F>
struct invocable_with_option :
       ::std::bool_constant<::std::is_invocable_v<F , name>              ||
                            ::std::is_invocable_v<F , width>             ||
                            ::std::is_invocable_v<F , precision>         ||
                            ::std::is_invocable_v<F , default_precision> ||
                            ::std::is_invocable_v<F , fill>              ||
                            ::std::is_invocable_v<F , fixed>             ||
                            ::std::is_invocable_v<F , unfixed>           ||
                            ::std::is_invocable_v<F , left>              ||
                            ::std::is_invocable_v<F , right>             ||
                            ::std::is_invocable_v<F , hex>               ||
                            ::std::is_invocable_v<F , decimal>           ||
                            ::std::is_invocable_v<F , octal>>
{};

template<typename F>
static constexpr bool invocable_with_option_v = invocable_with_option<F>::value;

template<typename T, typename = ::std::void_t<>>
struct has_value_field :
       ::std::false_type
{};

template<typename T>
struct has_value_field<T, ::std::void_t<decltype( ::std::declval<T>().value )>> :
       ::std::true_type
{};

template<typename T>
static constexpr bool has_value_field_v = has_value_field<T>::value;

template<typename T , typename = ::std::void_t<>>
struct is_input_iter : ::std::false_type
{};

template <typename T>
struct is_input_iter< T , ::std::void_t
                          <
                            decltype
                            (
                                ++::std::declval<T&>() ,
                                *::std::declval<T&>() ,
                                ::std::declval<T&>() == std::declval<T&>()
                            )
                           >
                    > : ::std::true_type
{};

template<typename T>
static constexpr bool is_input_iter_v = is_input_iter<T>::value;

template<typename Iter>
struct sequence
{
    Iter beg;
    Iter end;
};

template<typename T>
struct is_sequence : ::std::false_type
{};

template<typename Iter>
struct is_sequence<sequence<Iter>> : ::std::true_type
{};

template<typename T>
static constexpr bool is_sequence_v = is_sequence<T>::value;

template<typename T , typename = ::std::void_t<>>
struct is_container : ::std::false_type
{};

template<typename T>
struct is_container< T ,
                     ::std::void_t
                     <
                        decltype
                        (
                            ::std::begin( ::std::declval<T>() ) ,
                            ::std::end( ::std::declval<T>() )
                        )
                     >
                   > : ::std::true_type
{};

template<typename T>
static constexpr bool is_container_v = is_container<T>::value;

template<typename F>
inline void run( F val , const option& opt )
{
    static_assert(
        detail::invocable_with_option_v<F> ,
        "F must be invocable with an option."
    );

    visit(
        detail::overloaded {
            []( auto&& ) {} ,
            val
        } ,
        opt
    );
}

template<typename F>
inline void run( F val , const ::std::vector<option>& options )
{
    for ( const option& opt : options )
        run( val , opt );
}

template<typename Opt>
inline ::std::vector<option> filter_opts( const ::std::vector<option>& options )
{
    static_assert( is_option_v<Opt> , "T must be an option." );

    ::std::vector<option> filtered;

    for ( const option& opt : options )
    {
        visit(
            detail::overloaded {
                []( auto&& ) {} ,
                [ &filtered ] ( const Opt& opt )
                {
                    filtered.push_back( opt );
                }
            } ,
            opt
        );
    }

    return filtered;
}

inline ::std::string concat( const ::std::vector<option>& options )
{
    ::std::stringstream ss;

    ss << "[ ";

    for ( const option& opt : options )
    {
        visit(
            [ &ss ]( const auto& opt )
            {
                if constexpr ( has_value_field_v<decltype( opt )> )
                {
                    ss << '\'' << opt.value << "' ";
                }
            } ,
            opt
        );
    }

    ss << "]";

    return ss.str();
};

}

struct printer_exception : ::std::logic_error
{
    using ::std::logic_error::logic_error;
};

struct arguments_size_doesnt_match_with_columns : printer_exception
{
    using printer_exception::printer_exception;
};

struct duplicate_option : printer_exception
{
    using printer_exception::printer_exception;
};

struct both_left_and_right_option : printer_exception
{
    using printer_exception::printer_exception;
};

struct both_precision_and_default_precision : printer_exception
{
    using printer_exception::printer_exception;
};

struct both_fixed_and_unfixed : printer_exception
{
    using printer_exception::printer_exception;
};

template<typename Iter>
inline auto sequence( Iter beg , Iter end )
{
    static_assert(
        detail::is_input_iter_v<Iter> ,
        "'Iter' should support ++,*,!="
    );

    return detail::sequence<Iter> { beg , end };
}

template<typename Container>
inline auto sequence( Container&& c )
{
    static_assert(
        detail::is_container_v<Container> ,
        "'Container' should have 'begin()' and 'end()'"
    );

    using Iterator = decltype( ::std::begin( c ) );

    return detail::sequence<Iterator> {
        ::std::begin( c ) ,
        ::std::end( c )
    };
}

struct column
{
    column( ::std::initializer_list<option> opts )
    {
        copy(
            begin ( opts ) ,
            end( opts ) ,
            back_inserter( options )
        );
    }

    ::std::vector<option> options;
};

class printer
{
public:

    using osref = ::std::reference_wrapper<::std::ostream>;

    inline explicit printer(
        ::std::vector<column> columns ,
        ::std::vector<osref>  streams = ::std::vector<osref> {}
    );

    inline explicit printer(
        ::std::vector<column> columns ,
        ::std::ostream& stream
    );

    template<typename... Ts>
    inline printer& add_streams( ::std::ostream& os , Ts&... streams );

    template<typename... Ts>
    inline printer& remove_streams( ::std::ostream& os , Ts&... streams );

    template<typename... Ts>
    inline printer& print( const ::std::tuple<Ts...>& values );

    template<typename... Ts>
    inline printer& print( const Ts&... );
    inline printer& echo( ::std::string_view );
    inline printer& print_headers();
    inline const printer& sanity_check() const;
    inline printer& sanity_check();
    inline printer& flush();
    inline const ::std::vector<osref>& streams() const noexcept;

private:

    template<typename H , typename... T>
    inline void print_column( int col , ::std::ostream& , const H& , const T&... );

    template<typename... Ts , ::std::size_t... Idx>
    inline void print_tuple( const ::std::tuple<Ts...>& , ::std::index_sequence<Idx...> );

    template<typename T>
    inline void throw_if_duplicate_opt( const column& , ::std::string_view error ) const;

    inline void throw_if_both_left_and_right_opt( const column& ) const;
    inline void throw_if_both_precision_and_default_precision_opt( const column& ) const;
    inline void throw_if_both_fixed_and_unfixed_opt( const column& ) const;

    ::std::vector<osref> m_streams;
    ::std::vector<column> m_columns;
};

}

tableprinter::printer::printer( ::std::vector<column> columns , ::std::vector<osref> streams )
    :   m_columns { move( columns ) }
    ,   m_streams { move( streams ) }
{   }

tableprinter::printer::printer( ::std::vector<column> columns , ::std::ostream& stream )
    :   printer
        {
            move( columns ) ,
            ::std::vector<osref>{ stream }
        }
{   }

template<typename... Ts>
tableprinter::printer& tableprinter::printer::add_streams( std::ostream& os , Ts&... streams )
{
    static_assert(
        (::std::is_convertible_v<Ts& , ::std::ostream&> && ...) ,
        "Ts should be inherited from 'std::ostream'"
    );

    m_streams.push_back( os );
    ( m_streams.push_back( streams ) , ... );

    return *this;
}

template<typename... Ts>
tableprinter::printer& tableprinter::printer::remove_streams( ::std::ostream& os , Ts&... streams )
{
    static_assert(
        (::std::is_convertible_v<Ts& , ::std::ostream&> && ...) ,
        "Ts should be inherited from 'std::ostream'"
    );

    m_streams.erase(
        remove_if(
            begin( m_streams ) ,
            end( m_streams ) ,
            [ &os , &streams... ]( const osref& stream ){
                return &os == &stream.get() ||
                       ( ( addressof( streams ) == &stream.get() ) || ... );
            }
        ) ,
        end( m_streams )
    );

    return *this;
}

template<typename... Ts>
tableprinter::printer& tableprinter::printer::print( const Ts&... params )
{
    static_assert( sizeof...( Ts ) , "There must be arguments to print." );

    for ( ::std::ostream& stream : m_streams )
    {
        print_column<Ts...>( 0 , stream , params... );

        stream << '\n';
    }

    return *this;
}

template<typename... Ts>
tableprinter::printer& tableprinter::printer::print( const ::std::tuple<Ts...>& values )
{
    if ( auto count = sizeof...( Ts ) != size( m_columns ) )
        throw arguments_size_doesnt_match_with_columns {
            "There are " +
            ::std::to_string( size( m_columns ) ) +
            " columns but given " +
            ::std::to_string( count ) +
            " arguments."
        };

    print_tuple( values , ::std::make_index_sequence<sizeof...( Ts )>() );

    return *this;
}

tableprinter::printer& tableprinter::printer::echo( ::std::string_view str )
{
    for ( ::std::ostream& stream : m_streams )
        stream << str << "\n";

    return *this;
}

tableprinter::printer& tableprinter::printer::print_headers()
{
    for ( ::std::ostream& stream : m_streams )
    {
        for ( const auto& col : m_columns )
        {
            detail::run(
                detail::overloaded {
                    [ &stream ]( const width& w )
                    {
                        stream << ::std::setw( w );
                    } ,
                    [ &stream]( const left& )
                    {
                        stream << ::std::left;
                    } ,
                    [ &stream ]( const right& )
                    {
                        stream << ::std::right;
                    }
                } ,
                col.options
            );

            bool printed {};

            detail::run(
                [ &stream , &printed ]( const name& n )
                {
                    printed = true;
                    stream << n.value;
                } ,
                col.options
            );

            if ( !printed )
                stream << "";
        }

        stream << '\n';
    }

    return *this;
}

const tableprinter::printer& tableprinter::printer::sanity_check() const
{
    for ( const column& column : m_columns )
    {
        throw_if_duplicate_opt<name>( column , "Multiple 'name' options are not sane." );
        throw_if_duplicate_opt<width>( column , "Multiple 'width' options are not sane.");
        throw_if_duplicate_opt<precision>( column , "Multiple 'precision' options are not sane." );
        throw_if_duplicate_opt<fill>( column , "Multiple 'fill' options are not sane." );
        throw_if_both_left_and_right_opt( column );
        throw_if_both_precision_and_default_precision_opt( column );
        throw_if_both_fixed_and_unfixed_opt( column );
    }

    return *this;
}

tableprinter::printer& tableprinter::printer::sanity_check()
{
    const auto& self { *this };

    self.sanity_check();

    return *this;
}

tableprinter::printer& tableprinter::printer::flush()
{
    for ( ::std::ostream& os : m_streams )
        os.flush();

    return *this;
}

const ::std::vector<tableprinter::printer::osref>&
tableprinter::printer::streams() const noexcept
{
    return m_streams;
}

template<typename H , typename... T>
void tableprinter::printer::print_column( int col , ::std::ostream& os , const H& val , const T&... rest )
{
    if constexpr ( detail::is_sequence_v<H> )
    {
        for ( auto itr = val.beg; !( itr == val.end ); ++itr , ++col )
            print_column( col , os , *itr );
    }
    else
    {
        const auto& options { m_columns[ col++ ].options };

        detail::run(
            detail::overloaded {
                [ &os ]( const width& w )
                {
                    os << ::std::setw( w );
                } ,
                [ &os ]( const fill& f )
                {
                    os << ::std::setfill( f.value );
                } ,
                [ &os ]( const precision& p )
                {
                    os << ::std::setprecision( p );
                } ,
                [ &os ]( const default_precision& )
                {
                    os << ::std::setprecision( 6 );
                } ,
                [ &os ]( const fixed& f )
                {
                    os << ::std::fixed;
                } ,
                [ &os ]( const unfixed& )
                {
                    os.unsetf( ::std::ios_base::fixed );
                } ,
                [ &os ]( const left& )
                {
                    os << ::std::left;
                } ,
                [ &os ]( const right& )
                {
                    os << ::std::right;
                } ,
                [ &os ]( const hex& )
                {
                    os << ::std::hex;
                } ,
                [ &os ]( const decimal& )
                {
                    os << ::std::dec;
                } ,
                [ &os ]( const octal& )
                {
                    os << ::std::oct;
                }
            } ,
            options
        );

        os << val;
    }

    if constexpr ( bool( sizeof...( rest ) ) )
        print_column( col , os , rest... );
}

template<typename... Ts , ::std::size_t... Idx>
void tableprinter::printer::print_tuple( const ::std::tuple<Ts...>& values , ::std::index_sequence<Idx...> )
{
    print( ::std::get<Idx>( values )... );
}

template<typename T>
void tableprinter::printer::throw_if_duplicate_opt( const column& col , ::std::string_view error ) const
{
    auto opts = detail::filter_opts<T>( col.options );

    if ( size( opts ) > 1 )
    {
        ::std::string concatenated;

        throw duplicate_option {
            ::std::string { error } +
            " " +
            detail::concat( opts )
        };
    }
}

void tableprinter::printer::throw_if_both_left_and_right_opt( const column& col ) const
{
    auto lefts  = detail::filter_opts<left>( col.options );
    auto rights = detail::filter_opts<right>( col.options );

    if ( !empty( lefts ) && !empty( rights ) )
    {
        throw both_left_and_right_option {
            "Specifying both 'left' and 'right' options are not sane."
        };
    }
}

void tableprinter::printer::throw_if_both_precision_and_default_precision_opt( const column& col ) const
{
    auto precisions    = detail::filter_opts<precision>( col.options );
    auto default_precs = detail::filter_opts<default_precision>( col.options );

    if ( !empty( precisions ) && !empty( default_precs ) )
    {
        throw both_precision_and_default_precision {
            "Specifying both 'precision' and 'default_precision' options are not sane."
        };
    }
}

void tableprinter::printer::throw_if_both_fixed_and_unfixed_opt( const column& col ) const
{
    auto fixeds   = detail::filter_opts<fixed>( col.options );
    auto unfixeds = detail::filter_opts<unfixed>( col.options );

    if ( !empty( fixeds ) && !empty( unfixeds ) )
    {
        throw both_fixed_and_unfixed {
            "Specifying both 'fixed' and 'unfixed' options are not sane."
        };
    }
}