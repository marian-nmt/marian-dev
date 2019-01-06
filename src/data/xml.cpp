#include "data/corpus.h"

namespace marian {
namespace data {
namespace xml {

void processXml(const std::string& line,
                std::string& stripped_line,
                const Ptr<Vocab> target_vocab,
                SentenceTuple& tup) {
  Ptr<XmlOptions> xmlOptions = New<XmlOptions>();
  tup.setXmlOptions( xmlOptions );

  // no xml tag? we're done.
  if (line.find("<") == std::string::npos) {
    stripped_line = line;
    return;
  }

  // break up input into a vector of xml tags and text
  // example: (this), (<b>), (is a), (</b>), (test .)
  std::vector<std::string> xmlTokens = tokenizeXml(line);

  // we need to store opened tags, until they are closed
  // tags are stored as tripled (tagname, startpos, contents)
  typedef std::pair< std::string, std::pair< size_t, std::string > > OpenedTag;
  std::vector< OpenedTag > tagStack; // stack that contains active opened tags

  stripped_line = "";
  size_t wordPos = 0; // position in sentence (in terms of number of words)

  // loop through the tokens
  for (size_t xmlTokenPos=0 ; xmlTokenPos < xmlTokens.size() ; xmlTokenPos++) {

    // not a xml tag, but regular text (may contain many words)
    if (!isXmlTag(xmlTokens[xmlTokenPos])) {
      // add a space at boundary, if necessary
      if (stripped_line.size()>0 &&
          stripped_line[stripped_line.size() - 1] != ' ' &&
          xmlTokens[xmlTokenPos][0] != ' ') {
        stripped_line += " ";
      }
      stripped_line += xmlTokens[xmlTokenPos]; // add to output
      std::vector< std::string > outputWords; // ???
      utils::split(stripped_line, outputWords, " "); // ???
      wordPos = outputWords.size(); // count all the words // ???
    }

    // xml tag
    else {
      std::string tag = TrimXml(xmlTokens[xmlTokenPos]);
      utils::trim(tag);

      // tag without content ("<>", "< >", etc.)
      if (tag.size() == 0) {
        continue;
      }

      // check if unary (e.g., "<wall/>")
      bool isUnary = ( tag[tag.size() - 1] == '/' );

      // check if opening tag (e.g. "<a>", not "</a>")
      bool isClosed = ( tag[0] == '/' );
      bool isOpen = !isClosed;

      if (isClosed && isUnary) {
        std::cerr << "XML ERROR: can't have both closed and unary tag <" << tag << ">: " << line
                  << std::endl;
        continue;
      }

      if (isClosed)
        tag = tag.substr(1); // remove "/" at the beginning
      if (isUnary)
        tag = tag.substr(0,tag.size()-1); // remove "/" at the end

      // find the tag name and contents
      std::string::size_type endOfName = tag.find_first_of(' ');
      std::string tagName = tag;
      std::string tagContent = "";
      if (endOfName != std::string::npos) {
        tagName = tag.substr(0,endOfName);
        tagContent = tag.substr(endOfName+1);
      }

      // process new tag
      if (isOpen || isUnary) {
        OpenedTag openedTag = std::make_pair( tagName, std::make_pair( wordPos, tagContent ) );
        tagStack.push_back( openedTag );
      }

      // process completed tag
      if (isClosed || isUnary) {

        // pop last opened tag from stack;
        if (tagStack.size() == 0) {
          std::cerr << "XML ERROR: tag " << tagName << " closed, but not opened"
                    << ":" << line << std::endl;
          continue;
        }
        OpenedTag openedTag = tagStack.back();
        tagStack.pop_back();

        // tag names have to match
        if (openedTag.first != tagName) {
          std::cerr << "XML ERROR: tag " << openedTag.first << " closed by tag " << tagName << ": "
                    << line << std::endl;
          continue;
        }

        // assemble remaining information about tag
        size_t startPos = openedTag.second.first;
        std::string& tagContent = openedTag.second.second;
        size_t endPos = wordPos;

        // span attribute overwrites position
        std::string span = parseXmlTagAttribute(tagContent,"span");
        if (! span.empty()) {
          std::vector<std::string> ij;
          utils::split(span, ij, "-");
          if (ij.size() != 1 && ij.size() != 2) {
            std::cerr << "XML ERROR: span attribute must be of the form \"i-j\" or \"i\": " << line
                      << std::endl;
          }
          else {
            startPos = atoi(ij[0].c_str());
            if (ij.size() == 1) endPos = startPos + 1;
            else endPos = atoi(ij[1].c_str()) + 1;
          }
        }

        // TODO: special tag: wall
        if (0) {}
        // TODO: special tag: zone
        else if (0) {}

        // default: opening tag that specifies translation options
        else {
          if (startPos > endPos) {
            std::cerr << "XML ERROR: tag " << tagName << " startPos > endPos: " << line << std::endl;
            continue;
          }
          else if (startPos == endPos) {
            std::cerr << "XML ERROR: tag " << tagName << " span: " << line << std::endl;
            continue;
          }

          // specified translations -> vector of phrases
          std::string translation = parseXmlTagAttribute(tagContent,"translation");
          if (translation.empty()) {
            translation = parseXmlTagAttribute(tagContent,"english");
          }
          if (translation.empty()) {
            continue;
          }
          std::vector< std::string > translationWords;
          utils::trim(translation);
          utils::split(translation, translationWords, " ");
          std::cerr << "new option (" << startPos << "," << endPos << ") " << translation
                    << ", size " << translationWords.size() << "\n";

          // TODO: make sure that the third argument (inference) should be set to true
          Words translation_words = target_vocab->encode(translation, /* addEos= */ false, true);

          std::cerr << "encoded size " << translation_words.size() << "\n";
          std::cerr << "word id = " << translation_words[0] << "\n";
          Ptr<XmlOption> xmlOption = New<XmlOption>(startPos, endPos, translation_words);
          xmlOptions->push_back( xmlOption );
          Ptr<XmlOption> option = xmlOption;
          const Words &output = option->getOutput();
          std::cerr << "created XmlOption " << option << ": " << option->getStart() << "-"
                    << option->getEnd() << ", output length " << output.size() << "\n";
        }
      }
    }
  }
}

/**
 * Remove "<" and ">" from XML tag
 */
std::string TrimXml(const std::string& str) {
  // too short to be xml token -> do nothing
  if (str.size() < 2) return str;

  // strip first and last character
  if (isXmlTag(str)) {
    return str.substr(1, str.size()-2);
  }
  // not an xml token -> do nothing
  else {
    return str;
  }
}

/**
 * Check if the token is an XML tag, i.e. starts with "<"
 *
 * \param tag token to be checked
 * \param lbrackStr xml tag's left bracket string, typically "<"
 * \param rbrackStr xml tag's right bracket string, typically ">"
 */
bool isXmlTag(const std::string& tag) {
  return tag[0] == '<' &&
         tag[tag.size()-1] == '>' &&
         ( tag[1] == '/'
           || (tag[1] >= 'a' && tag[1] <= 'z')
           || (tag[1] >= 'A' && tag[1] <= 'Z') );
}

/**
 * @brief Get value for XML attribute, if it exists
 */
std::string parseXmlTagAttribute(const std::string& tag, const std::string& attributeName) {
  std::string tagOpen = attributeName + "=\"";
  size_t contentsStart = tag.find(tagOpen);
  if (contentsStart == std::string::npos) return "";
  contentsStart += tagOpen.size();
  size_t contentsEnd = tag.find_first_of('"',contentsStart+1);
  if (contentsEnd == std::string::npos) {
    std::cerr << "XML ERROR: Malformed XML attribute: "<< tag << std::endl;
    return "";
  }
  size_t possibleEnd;
  while (tag.at(contentsEnd-1) == '\\' && (possibleEnd = tag.find_first_of('"',contentsEnd+1)) != std::string::npos) {
    contentsEnd = possibleEnd;
  }
  return tag.substr(contentsStart,contentsEnd-contentsStart);
}

/**
 * @brief Code to break up a xml-tagged input sentence into tokens
   and tags. Adapted from Moses.
 */
std::vector<std::string> tokenizeXml(const std::string& line) {

  std::vector<std::string> tokens; // vector of tokens to be returned
  std::string::size_type cpos = 0; // current position in string
  std::string::size_type lpos = 0; // left start of xml tag
  std::string::size_type rpos = 0; // right end of xml tag

  // walk thorugh the string (loop vver cpos)
  while (cpos != line.size()) {

    // find the next opening "<" of an xml tag
    lpos = line.find("<", cpos);
    if (lpos != std::string::npos) {
      // find the end of the xml tag
      rpos = line.find(">", lpos);
      // sanity check: there has to be closing ">"
      if (rpos == std::string::npos) {
        std::cerr << "XML ERROR: malformed XML: " << line << std::endl;
        return tokens;
      }
    }
    else { // no more tags found
      // add the rest as token
      tokens.push_back(line.substr(cpos));
      break;
    }

    // add stuff before xml tag as token, if there is any
    if (lpos - cpos > 0)
      tokens.push_back(line.substr(cpos, lpos - cpos));

    // add xml tag as token
    tokens.push_back(line.substr(lpos, rpos-lpos+1));
    cpos = rpos + 1;
  }
  return tokens;
}

}  // namespace xml
}  // namespace data
}  // namespace marian
